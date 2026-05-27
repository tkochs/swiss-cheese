use super::utils;
use crate::utils::{StringEncoding, arr_to_out, pyany_to_vec};
use ndarray::Array2;
use ndarray_stats::CorrelationExt;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::prelude::*;
use std::sync::{Arc, mpsc::channel};
use threadpool::ThreadPool;

#[pyclass(name = "MAR")]
pub struct MAR {
    rng: StdRng,
    pool: ThreadPool,
}

#[pymethods]
impl MAR {
    #[new]
    #[pyo3(signature = (seed=None, n_workers=None))]
    fn new(seed: Option<u64>, n_workers: Option<usize>) -> MAR {
        let seed = seed.unwrap_or(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("Getting time failed")
                .as_nanos() as u64,
        );
        let rng = StdRng::seed_from_u64(seed);
        let pool = ThreadPool::new(n_workers.unwrap_or(4));
        MAR { rng, pool }
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
        let alpha = adjust_alpha(&arr, alpha);
        let n_missing = (arr.len() as f64 * alpha).ceil() as usize;
        let (miss_cols, pairs) = self.pairs(arr, alpha);
        let mut missing_count = 0;
        while missing_count < n_missing {
            let cols = select_cols(&mut self.rng, arr, missing_count, n_missing, &miss_cols);
            missing_count += cols.len();
        }
    }

    fn drop_cols(&mut self, arr: &mut Arc<Array2<f64>>, cols: &[usize]) {
        let (transmitter, receiver) = channel();
        for &c in cols {
            let transmitter = transmitter.clone();
            let _arr = Arc::clone(&arr);
            self.pool.execute(move || {
                let i = _arr
                    .column(c)
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| !v.is_nan())
                    .min_by(|(_, a), (_, b)| (*a).total_cmp(b))
                    .expect("No argmin found!")
                    .0;
                transmitter.send((i, c)).unwrap();
            });
        }
        drop(transmitter);
        self.pool.join();
        let indices: Vec<_> = receiver.iter().collect();
        let arr = Arc::get_mut(arr).expect("Still references alive");
        for (r, c) in indices {
            arr[(r, c)] = f64::NAN;
        }
    }

    fn pairs(&mut self, arr: &Arc<Array2<f64>>, alpha: f64) -> (Vec<usize>, Vec<(usize, usize)>) {
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
        (
            miss_cols.to_vec(),
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
        return (0..ncols).collect();
    }
    (0..ncols).sample(rng, n_missing - count)
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
        let _ = MAR::new(None, None);
        let _ = MAR::new(Some(5), Some(1));
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
