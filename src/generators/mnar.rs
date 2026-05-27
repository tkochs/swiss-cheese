use super::utils::{Gauss, fix};
use crate::utils::{StringEncoding, arr_to_out, pyany_to_vec};
use ndarray::Array2;
use pyo3::prelude::*;
use rand::prelude::*;
use std::sync::{Arc, mpsc::channel};
use threadpool::ThreadPool;

#[pyclass(name = "MNARrs")]
pub struct MNAR {
    mean: f64,
    variance: f64,
    rng: StdRng,
    pool: ThreadPool,
}

#[pymethods]
impl MNAR {
    #[new]
    #[pyo3(signature = (mean= None, variance=None, seed=None, n_workers=None))]
    fn new(
        mean: Option<f64>,
        variance: Option<f64>,
        seed: Option<u64>,
        n_workers: Option<usize>,
    ) -> MNAR {
        let mean = mean.unwrap_or(0.5);
        let variance = variance.unwrap_or(0.0);
        let seed = seed.unwrap_or(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("Getting time failed")
                .as_nanos() as u64,
        );
        let rng = StdRng::seed_from_u64(seed);
        let pool = ThreadPool::new(n_workers.unwrap_or(4));
        MNAR {
            mean,
            variance,
            rng,
            pool,
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
        fix();
        let array = ndarray::Array2::from_shape_vec((nrows, ncols), vec)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let mut arr = Arc::new(array);
        self.drop(&mut arr, alpha);
        arr_to_out(py, &arr, out, enc_info)
    }

    fn __repr__(&self) -> String {
        format!("MNAR[{}]", self.mean)
    }
}

impl MNAR {
    fn drop(&mut self, arr: &mut Arc<Array2<f64>>, alpha: f64) {
        let n_missing = (arr.len() as f64 * alpha).ceil() as usize;
        let mut missing_count = 0;
        let distributions = get_distribution(self.mean, self.variance, &arr);
        while missing_count < n_missing {
            let cols = select_cols(&mut self.rng, arr, missing_count, n_missing);
            self.drop_cols(arr, &distributions, &cols);
            missing_count += cols.len();
            println!("{:?}", arr);
        }
    }

    fn drop_cols(&mut self, arr: &mut Arc<Array2<f64>>, distributions: &[Gauss], cols: &[usize]) {
        let (transmitter, receiver) = channel();
        let samples: Vec<f64> = cols
            .iter()
            .map(|&c| distributions[c].sample(&mut self.rng))
            .collect();
        for (&c, &s) in cols.iter().zip(&samples) {
            assert!(!s.is_nan(), "Sample is nan");
            let transmitter = transmitter.clone();
            let _arr = Arc::clone(&arr);
            self.pool.execute(move || {
                let i = _arr
                    .column(c)
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| !v.is_nan())
                    .min_by(|(_, a), (_, b)| {
                        ((*a - s) * (*a - s)).total_cmp(&((*b - s) * (*b - s)))
                    })
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
}

fn select_cols(rng: &mut StdRng, arr: &Array2<f64>, count: usize, n_missing: usize) -> Vec<usize> {
    let ncols = arr.ncols();
    if n_missing - count >= ncols {
        return (0..ncols).collect();
    }
    (0..ncols).sample(rng, n_missing - count)
}

fn get_distribution(mean: f64, var: f64, arr: &Array2<f64>) -> Vec<Gauss> {
    let mut dist = Vec::with_capacity(arr.ncols());
    let mut buff = Vec::with_capacity(arr.nrows());
    for i in 0..arr.ncols() {
        let col = arr.column(i);
        let (local_mean, local_var) = transform(&mut buff, col.iter().copied(), mean, var);
        dist.push(Gauss::new(local_mean, local_var));
    }
    dist
}

fn transform(
    buff: &mut Vec<f64>,
    col: impl Iterator<Item = f64>,
    mean: f64,
    var: f64,
) -> (f64, f64) {
    buff.clear();
    buff.extend(col);
    let n = (buff.len() as f64 * mean).floor() as usize;
    buff.sort_unstable_by(|a, b| a.total_cmp(b));
    let q = if buff.len() % 2 == 1 || n == 0 {
        buff[n]
    } else {
        (buff[n - 1] + buff.get(n).unwrap_or_else(|| &buff[n - 1])) / 2.0
    };
    (
        q,
        (buff.last().expect("No elements in col") - buff[0]) * var,
    )
}

#[cfg(test)]
mod test {
    use crate::generators::mnar::transform;

    use super::*;

    #[test]
    fn create() {
        let _ = MNAR::new(None, None, None, None);
        let _ = MNAR::new(Some(0.5), Some(1.0), None, None);
    }

    #[test]
    fn _transform() {
        let v = [1.0, 1.0, 2.0, 3.0, 4.0, 5.0].iter().copied();
        let mut buff: Vec<f64> = vec![0.0; v.len()];
        let (mean, var) = transform(&mut buff, v, 0.5, 1.0);
        assert!((mean - 2.5).abs() < 1e-17, "Actual Mean: {mean}");
        assert!((var - 4.0).abs() < 1e-17, "Actual Var: {var}");
        let v = [0.0, 1.0, 2.0, 3.0, 4.0].iter().copied();
        let mut buff: Vec<f64> = vec![0.0; v.len()];
        let (mean, var) = transform(&mut buff, v, 0.25, 0.0);
        assert!((mean - 1.0).abs() < 1e-17, "Actual Mean: {mean}");
        assert!((var - 0.0).abs() < 1e-17, "Actual Var: {var}");
        let v: Vec<f64> = (0..10).map(|x| x as f64).collect();
        let mut buff: Vec<f64> = vec![0.0; v.len()];
        let (mean, var) = transform(&mut buff, v.iter().copied(), 0.5, 0.0);
        assert!((mean - 4.5).abs() < 1e-17, "Actual Mean: {mean}");
        assert!((var - 0.0).abs() < 1e-17, "Actual Var: {var}");
        let (mean, var) = transform(&mut buff, v.iter().copied(), 0.0, 1.0);
        assert!((mean - 0.0).abs() < 1e-17, "Actual Mean: {mean}");
        assert!((var - 9.0).abs() < 1e-17, "Actual Var: {var}");
        let (mean, var) = transform(&mut buff, v.iter().copied(), 1.0, 0.0);
        assert!((mean - 9.0).abs() < 1e-17, "Actual Mean: {mean}");
        assert!((var - 0.0).abs() < 1e-17, "Actual Var: {var}");
    }
}
