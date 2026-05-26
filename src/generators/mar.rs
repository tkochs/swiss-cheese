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
    ratio: f64,
    rng: StdRng,
    pool: ThreadPool,
}

#[pymethods]
impl MAR {
    #[new]
    #[pyo3(signature = (ratio= None, seed=None, n_workers=None))]
    fn new(ratio: Option<f64>, seed: Option<u64>, n_workers: Option<usize>) -> MAR {
        let ratio = ratio.unwrap_or(0.5);
        let seed = seed.unwrap_or(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("Getting time failed")
                .as_nanos() as u64,
        );
        let rng = StdRng::seed_from_u64(seed);
        let pool = ThreadPool::new(n_workers.unwrap_or(4));
        MAR { ratio, rng, pool }
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
        format!("MAR[{}]", self.ratio)
    }
}

impl MAR {
    fn drop(&mut self, arr: &mut Arc<Array2<f64>>, alpha: f64) {
        let n_missing = (arr.len() as f64 * alpha).ceil() as usize;
        let available = self.pairs(arr);
        let mut missing_count = 0;
        while missing_count < n_missing {
            let cols = select_cols(&mut self.rng, arr, missing_count, n_missing, &available);
            missing_count += cols.len();
            println!("{:?}", arr);
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

    fn pairs(&mut self, arr: &Arc<Array2<f64>>) -> Vec<(usize, usize)> {
        let (miss_cols, obs_cols): (Vec<_>, Vec<_>) =
            (0..arr.ncols()).partition(|_| self.rng.random_bool(self.ratio));
        let correlations = arr
            .t()
            .pearson_correlation()
            .expect("Failed to calculate pearson_correlation");
        let pair = miss_cols.iter().map(|c| correlations.column(*c));
        Vec::new()
    }
}

fn select_cols(
    rng: &mut StdRng,
    arr: &Array2<f64>,
    count: usize,
    n_missing: usize,
    available: &[(usize, usize)],
) -> Vec<usize> {
    let ncols = arr.ncols();
    if n_missing - count >= ncols {
        return (0..ncols).collect();
    }
    (0..ncols).sample(rng, n_missing - count)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn create() {
        let _ = MAR::new(None, None, None);
        let _ = MAR::new(Some(0.5), Some(1), None);
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
