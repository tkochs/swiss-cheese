use super::{common, common::Generator, common::mode::*, constants, utils};
use crate::generators::utils::{Gauss, get_distribution};
use ndarray::Array2;
use ndarray_stats::CorrelationExt;
use pyo3::prelude::*;
use rand::prelude::*;
use rayon::prelude::*;

#[pyclass(name = "MAR")]
pub struct MAR {
    max_missing_per_column: f64,
    rng: StdRng,
    mode: Mode,
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
    ) -> PyResult<MAR> {
        let mode: Mode = Mode::new(mode, mean, variance, None, None)?;
        let rng = common::build_rng(random_seed);
        if let Mode::BLOCK { .. } | Mode::BLOB(_) = mode {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Mode {} not supported for MAR yet.",
                mode.as_str(),
            )));
        }
        Ok(MAR {
            max_missing_per_column,
            rng,
            mode,
        })
    }

    fn __call__<'py>(
        &mut self,
        py: Python<'py>,
        data: &Bound<'_, PyAny>,
        missing_rate: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.call(py, data, missing_rate)
    }

    fn __repr__(&self) -> String {
        match self.mode {
            Mode::GM { mean, .. } => format!("MAR[{}]", mean),
            Mode::MIN => format!("MAR[MIN]"),
            Mode::MAX => format!("MAR[MAX]"),
            _ => panic!("No valid mode passed during construction!"),
        }
    }
}

impl Generator for MAR {
    fn max_missing_per_column(&self) -> f64 {
        self.max_missing_per_column
    }

    fn drop(&mut self, arr: &mut Array2<f64>, alpha: f64) {
        let n_missing = (arr.len() as f64 * alpha).ceil() as usize;
        let mut missing_count = 0;
        let (miss_cols, pairs) = self.pairs(arr, alpha);
        let (distributions, cmp): (
            Option<Vec<Gauss>>,
            fn(&f64, &f64, &f64) -> std::cmp::Ordering,
        ) = match self.mode {
            Mode::MAX => (None, |a, b, _| b.total_cmp(a)),
            Mode::MIN => (None, |a, b, _| a.total_cmp(b)),
            Mode::GM { mean, var } => (Some(get_distribution(mean, var, arr.view())), |a, b, s| {
                ((*a - s) * (*a - s)).total_cmp(&((*b - s) * (*b - s)))
            }),
            Mode::BLOCK { .. } | Mode::BLOB(_) => {
                panic!("Mode {} not supported for MAR yet.", self.mode.as_str())
            }
        };
        while missing_count < n_missing {
            let max_miss = (n_missing - missing_count).min(miss_cols.len());
            let samples = utils::get_samples(&distributions, &miss_cols, &mut self.rng);
            missing_count += self.drop_cols(arr, &samples[..max_miss], &pairs, cmp);
        }
    }
}

impl MAR {
    fn drop_cols(
        &mut self,
        arr: &mut Array2<f64>,
        samples: &[f64],
        // obs: &HashMap<usize, usize>,
        cols: &[(usize, usize)],
        cmp: fn(&f64, &f64, &f64) -> std::cmp::Ordering,
    ) -> usize {
        let indices: Vec<_> = cols
            .par_iter()
            .zip(samples)
            .filter_map(|(&c, &s)| {
                assert!(!s.is_nan(), "Sample is nan");
                arr.column(c.1)
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !arr[(*i, c.0)].is_nan())
                    .min_by(|(_, a), (_, b)| {
                        cmp(*a, *b, &s)
                        // ((*a - s) * (*a - s)).total_cmp(&((*b - s) * (*b - s)))
                    })
                    .map(|(i, _)| (i, c.0))
            })
            .collect();
        common::remove(arr, &indices)
    }

    fn pairs(
        &mut self,
        arr: &Array2<f64>,
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

fn _select_cols(
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
