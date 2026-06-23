use super::{common::mode::*, constants, utils};
use crate::utils::{StringEncoding, arr_to_out, pyany_to_vec};
use ndarray::Array2;
use ndarray_stats::CorrelationExt;
use pyo3::exceptions::PyUserWarning;
use pyo3::prelude::*;
use rand::prelude::*;
use rayon::prelude::*;
use std::sync::Arc;

#[pyclass(name = "MAR")]
pub struct MAR {
    max_missing_per_column: f64,
    rng: StdRng,
    mode: Mode,
    mean: f64,
    variance: f64,
}

#[pymethods]
impl MAR {
    #[new]
    #[pyo3(signature = (mean=None, variance=None, max_missing_per_column=constants::MAX_MISSING_PER_COLUMN, mode="GM", random_seed=None))]
    fn new(
        mean: Option<f64>,
        variance: Option<f64>,
        max_missing_per_column: f64,
        mode: &str,
        random_seed: Option<u64>,
    ) -> MAR {
        let mut r = rand::rng();
        let seed = random_seed.unwrap_or(r.random());
        let rng = StdRng::seed_from_u64(seed);
        MAR {
            max_missing_per_column,
            rng,
            mode: mode.into(),
            mean: mean.unwrap_or(0.5),
            variance: variance.unwrap_or(0.0),
        }
    }

    fn __call__<'py>(
        &mut self,
        py: Python<'py>,
        data: &Bound<'_, PyAny>,
        missing_rate: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (array, out, enc_info) = pyany_to_vec(data, &Some(StringEncoding::LabelEncoding))?;
        utils::fix();
        let mut arr = Arc::new(array);
        let missing_rate = self._adjust_alpha(py, arr.ncols(), missing_rate);
        self.drop(&mut arr, missing_rate);
        arr_to_out(py, &arr, out, enc_info)
    }

    fn __repr__(&self) -> String {
        format!("MAR")
    }

    #[inline]
    fn _adjust_alpha<'py>(&self, py: Python<'py>, n_cols: usize, alpha: f64) -> f64 {
        let max = self.max_missing_per_column - (self.max_missing_per_column / n_cols as f64);
        if alpha > max {
            let msg = std::ffi::CString::new(format!(
                "Warning: Missing rate too high to ensure MAR properties! Maximum missing rate: {}",
                max
            ))
            .unwrap();
            PyErr::warn(py, &py.get_type::<PyUserWarning>(), &msg, 0)
                .expect("Something went wrong..");
            max
        } else {
            alpha
        }
    }
}

impl MAR {
    fn drop(&mut self, arr: &mut Arc<Array2<f64>>, alpha: f64) {
        let n_missing = (arr.len() as f64 * alpha).ceil() as usize;
        let mut missing_count = 0;
        let (miss_cols, pairs) = self.pairs(arr, alpha);
        let distributions = utils::get_distribution(self.mean, self.variance, arr.view());
        let cmp: fn(&f64, &f64, &f64) -> std::cmp::Ordering = match self.mode {
            Mode::MAX => |a, b, _| a.total_cmp(b),
            Mode::MIN => |a, b, _| b.total_cmp(a),
            Mode::GM => |a, b, s| ((*a - s) * (*a - s)).total_cmp(&((*b - s) * (*b - s))),
        };
        while missing_count < n_missing {
            let cols = select_cols(&mut self.rng, arr, missing_count, n_missing, &miss_cols);
            self.drop_cols(arr, &distributions, &pairs, cmp);
            missing_count += cols.len();
        }
    }

    fn drop_cols(
        &mut self,
        arr: &mut Arc<Array2<f64>>,
        distributions: &[utils::Gauss],
        // obs: &HashMap<usize, usize>,
        cols: &[(usize, usize)],
        cmp: fn(&f64, &f64, &f64) -> std::cmp::Ordering,
    ) {
        let samples: Vec<f64> = cols
            .iter()
            .map(|&c| distributions[c.1].sample(&mut self.rng))
            .collect();
        let indices: Vec<_> = cols
            .par_iter()
            .zip(&samples)
            .map(|(&c, &s)| {
                assert!(!s.is_nan(), "Sample is nan");
                let i = arr
                    .column(c.1)
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !arr[(*i, c.0)].is_nan())
                    .min_by(|(_, a), (_, b)| {
                        cmp(*a, *b, &s)
                        // ((*a - s) * (*a - s)).total_cmp(&((*b - s) * (*b - s)))
                    })
                    .expect("No argmin found!")
                    .0;
                (i, c)
            })
            .collect();
        let arr = Arc::get_mut(arr).expect("Err: Multithreading issue detected!");
        for (r, c) in indices {
            arr[(r, c.0)] = f64::NAN;
        }
    }

    fn pairs(
        &mut self,
        arr: &Arc<Array2<f64>>,
        alpha: f64,
    ) -> (
        Vec<usize>, //HashMap<usize, usize>,
        Vec<(usize, usize)>,
    ) {
        let n_miss = f64::ceil(arr.ncols() as f64 * alpha / self.max_missing_per_column) as usize;
        let mut cols: Vec<_> = (0..arr.ncols()).collect();
        cols.shuffle(&mut self.rng);
        let (miss_cols, obs_cols) = cols.split_at(n_miss);

        let correlations = arr
            .t()
            .pearson_correlation()
            .expect("Failed to calculate pearson_correlation");
        let mut max_corr = vec![(0.0, 0); n_miss];
        for (&mc, id) in miss_cols.iter().zip(0..) {
            for &oc in obs_cols {
                let c = correlations[(mc, oc)];
                if c.abs() > max_corr[id].0 {
                    max_corr[id] = (c.abs(), oc);
                }
            }
        }
        // let mut obs = HashMap::with_capacity(obs_cols.len());
        // for (&o, id) in obs_cols.iter().zip(0..) {
        //     obs.insert(id as usize, o);
        // }
        (
            miss_cols.to_vec(),
            // obs,
            miss_cols
                .iter()
                .zip(max_corr)
                .map(|(a, (_, b))| (*a, b))
                .collect(),
        )
    }
}

fn select_cols(
    rng: &mut StdRng,
    arr: &Array2<f64>,
    count: usize,
    n_missing: usize,
    available: &[usize],
) -> Vec<usize> {
    let ncols = arr.ncols();

    if n_missing - count >= ncols {
        return (available).to_vec();
    }
    available
        .sample(rng, n_missing - count)
        .map(|x| *x)
        .collect()
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn create() {
        let _ = MAR::new(None, None, constants::MAX_MISSING_PER_COLUMN, "GM", None);
        let _ = MAR::new(None, None, constants::MAX_MISSING_PER_COLUMN, "GM", Some(5));
    }
    #[test]
    fn correlations() {
        // 4 samples (rows), 2 variables (columns)
        let data = Array2::from_shape_vec((4, 2), vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0])
            .unwrap();
        let corr_no_t = data.pearson_correlation().unwrap();
        let corr_t = data.t().pearson_correlation().unwrap();
        println!("Without .t(): {:?}", corr_no_t);
        println!("With .t(): {:?}", corr_t);
        assert!(corr_t.ncols() == 2 && corr_t.nrows() == 2);
        assert!(corr_no_t.ncols() == 4 && corr_no_t.nrows() == 4);
    }
}
