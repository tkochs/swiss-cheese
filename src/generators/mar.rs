use super::{common::mode::*, constants, utils};
use crate::utils::{StringEncoding, arr_to_out, pyany_to_vec};
use ndarray::Array2;
use ndarray_stats::CorrelationExt;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, mpsc::channel};
use threadpool::ThreadPool;

#[pyclass(name = "MAR")]
pub struct MAR {
    max_missing_per_column: f64,
    rng: StdRng,
    pool: ThreadPool,
    mode: Mode,
    mean: f64,
    variance: f64,
}

#[pymethods]
impl MAR {
    #[new]
    #[pyo3(signature = (mean=None, variance=None, max_missing_per_column=constants::MAX_MISSING_PER_COLUMN, mode="GM", seed=None, n_workers=None))]
    fn new(
        mean: Option<f64>,
        variance: Option<f64>,
        max_missing_per_column: f64,
        mode: &str,
        seed: Option<u64>,
        n_workers: Option<usize>,
    ) -> MAR {
        let mut r = rand::rng();
        let seed = seed.unwrap_or(r.random());
        let rng = StdRng::seed_from_u64(seed);
        let pool = ThreadPool::new(n_workers.unwrap_or(constants::N_WORKERS));
        MAR {
            max_missing_per_column,
            rng,
            pool,
            mode: mode.into(),
            mean: mean.unwrap_or(0.5),
            variance: variance.unwrap_or(0.0),
        }
    }

    fn __call__<'py>(
        &mut self,
        py: Python<'py>,
        data: &Bound<'_, PyAny>,
        alpha: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let ((vec, nrows, ncols), out, enc_info) =
            pyany_to_vec(py, data, Some(StringEncoding::LabelEncoding))?;
        utils::fix();
        let array = Array2::from_shape_vec((nrows, ncols), vec)
            .map_err(|e| PyErr::new::<PyValueError, _>(e.to_string()))?;
        let mut arr = Arc::new(array);
        self.drop(&mut arr, alpha);
        arr_to_out(py, &arr, out, enc_info)
    }

    fn __repr__(&self) -> String {
        format!("MAR")
    }
}

impl MAR {
    fn drop(&mut self, arr: &mut Arc<Array2<f64>>, alpha: f64) {
        let n_missing = (arr.len() as f64 * alpha).ceil() as usize;
        let mut missing_count = 0;
        let (miss_cols, obs_cols, pairs) = self.pairs(arr, alpha);
        let distributions = utils::get_distribution(self.mean, self.variance, arr.view());
        let cmp: fn(&f64, &f64, &f64) -> std::cmp::Ordering = match self.mode {
            Mode::MAX => |a, b, _| a.total_cmp(b),
            Mode::MIN => |a, b, _| b.total_cmp(a),
            Mode::GM => |a, b, s| ((*a - s) * (*a - s)).total_cmp(&((*b - s) * (*b - s))),
        };
        while missing_count < n_missing {
            let cols = select_cols(&mut self.rng, arr, missing_count, n_missing, &miss_cols);
            self.drop_cols(arr, &distributions, &obs_cols, &pairs, cmp);
            missing_count += cols.len();
        }
    }

    fn drop_cols(
        &mut self,
        arr: &mut Arc<Array2<f64>>,
        distributions: &[utils::Gauss],
        obs: &HashMap<usize, usize>,
        cols: &[(usize, usize)],
        cmp: fn(&f64, &f64, &f64) -> std::cmp::Ordering,
    ) {
        let (transmitter, receiver) = channel();
        let samples: Vec<f64> = cols
            .iter()
            .map(|&c| distributions[obs[&c.1]].sample(&mut self.rng))
            .collect();
        for (&c, &s) in cols.iter().zip(&samples) {
            assert!(!s.is_nan(), "Sample is nan");
            let transmitter = transmitter.clone();
            let _arr = Arc::clone(&arr);
            self.pool.execute(move || {
                let i = _arr
                    .column(c.1)
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| !v.is_nan())
                    .min_by(|(_, a), (_, b)| {
                        cmp(*a, *b, &s)
                        // ((*a - s) * (*a - s)).total_cmp(&((*b - s) * (*b - s)))
                    })
                    .expect("No argmin found!")
                    .0;
                transmitter.send((i, c)).unwrap();
            });
        }
        drop(transmitter);
        self.pool.join();
        let indices: Vec<_> = receiver.iter().collect();
        let arr = Arc::get_mut(arr).expect("Err: Multithreading issue detected!");
        for (r, c) in indices {
            arr[(r, c.0)] = f64::NAN;
        }
    }

    fn pairs(
        &mut self,
        arr: &Arc<Array2<f64>>,
        alpha: f64,
    ) -> (Vec<usize>, HashMap<usize, usize>, Vec<(usize, usize)>) {
        let n_miss = f64::ceil(arr.ncols() as f64 * alpha) as usize;
        assert!(
            n_miss < arr.ncols(),
            "All columns are targeted, missingness too high for MAR. Please fix."
        );

        let mut cols: Vec<_> = (0..arr.ncols()).collect();
        cols.shuffle(&mut self.rng);
        let (miss_cols, obs_cols) = cols.split_at(n_miss);

        let correlations = arr
            .t()
            .pearson_correlation()
            .expect("Failed to calculate pearson_correlation");
        let mut max_corr = vec![(0.0, 0); n_miss];
        for &mc in miss_cols {
            for &oc in obs_cols {
                let c = correlations[(mc, oc)];
                if c.abs() > max_corr[mc].0 {
                    max_corr[mc] = (c.abs(), oc);
                }
            }
        }
        let mut obs = HashMap::with_capacity(obs_cols.len());
        for (&o, id) in obs_cols.iter().zip(0..) {
            obs.insert(id as usize, o);
        }
        (
            miss_cols.to_vec(),
            obs,
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

#[inline]
fn adjust_alpha(arr: &Array2<f64>, alpha: f64) -> f64 {
    let max = 1.0 - (1.0 / arr.ncols() as f64);
    if alpha > max { max } else { alpha }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn create() {
        let _ = MAR::new(
            None,
            None,
            constants::MAX_MISSING_PER_COLUMN,
            "GM",
            None,
            None,
        );
        let _ = MAR::new(
            None,
            None,
            constants::MAX_MISSING_PER_COLUMN,
            "GM",
            Some(5),
            Some(1),
        );
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
