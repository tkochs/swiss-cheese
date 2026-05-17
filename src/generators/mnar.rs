use super::utils;
use crate::utils::{arr_to_out, pyany_to_vec};
use ndarray::Array2;
use ndarray_stats::QuantileExt;
// use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;
use rand::prelude::*;
// use rand::{RngExt, SeedableRng, rngs::StdRng, seq::IteratorRandom};

#[pyclass]
pub struct MNAR {
    mean: f64,
    variance: f64,
    rng: StdRng,
}

#[pymethods]
impl MNAR {
    #[new]
    #[pyo3(signature = (mean= None,variance=None, seed=None))]
    fn new(mean: Option<f64>, variance: Option<f64>, seed: Option<u64>) -> MNAR {
        let mean = mean.unwrap_or(0.5);
        let variance = variance.unwrap_or(0.0);
        let seed = seed.unwrap_or(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("Getting time failed")
                .as_nanos() as u64,
        );
        let rng = StdRng::seed_from_u64(seed);
        MNAR {
            mean,
            variance,
            rng,
        }
    }

    fn __call__<'py>(
        &mut self,
        py: Python<'py>,
        data: &Bound<'_, PyAny>,
        alpha: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let ((vec, nrows, ncols), out) = pyany_to_vec(py, data)?;
        utils::fix();
        let mut array = ndarray::Array2::from_shape_vec((nrows, ncols), vec)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let missing = self.drop(&mut array, alpha);
        arr_to_out(py, &missing, out)
    }
}

impl MNAR {
    fn drop(&mut self, arr: &mut Array2<f64>, alpha: f64) -> Array2<f64> {
        let n_missing = (arr.len() as f64 * alpha).ceil() as usize;
        let mut missing_count = 0;
        let distributions = get_distribution(&mut self.rng, self.mean, self.variance, &arr);
        while missing_count < n_missing {
            let cols = select_cols(&mut self.rng, &arr, missing_count, n_missing);
            drop_cols(arr, &distributions, &cols);
            missing_count += cols.len();
        }
        arr.to_owned()
    }
}

fn drop_cols(arr: &mut Array2<f64>, distributions: &Vec<Gauss>, cols: &Vec<usize>) {
    for c in cols {
        let s = distributions[*c].sample();
        let distances = (&arr.column(*c) - s).pow2();
        let i = distances.argmin_skipnan().expect("No argmin found!");
        arr[(i, *c)] = f64::NAN;
    }
}

fn select_cols(rng: &mut StdRng, arr: &Array2<f64>, count: usize, n_missing: usize) -> Vec<usize> {
    let ncols = arr.ncols();
    if n_missing - count >= ncols {
        return (0..ncols).collect();
    }
    (0..ncols).sample(rng, n_missing - count)
}

fn get_distribution(rng: &mut StdRng, mean: f64, var: f64, arr: &Array2<f64>) -> Vec<Gauss> {
    let mut dist = Vec::with_capacity(arr.ncols());
    for i in 0..arr.ncols() {
        let col = arr.column(i).to_owned();
        let (local_mean, local_var) = transform(
            col.as_slice().expect("Column not contigous in mem"),
            mean,
            var,
        );
        dist.push(Gauss::new(rng.random(), local_mean, local_var));
    }
    dist
}

fn transform(col: &[f64], mean: f64, var: f64) -> (f64, f64) {
    let n = (col.len() as f64 * mean).floor() as usize;
    let mut sorted = col.to_owned();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let q = if sorted.len() % 2 == 1 || n == 0 {
        sorted[n]
    } else {
        (sorted[n - 1] + sorted.get(n).unwrap_or_else(|| &sorted[n - 1])) / 2.0
    };
    (
        q,
        (sorted.last().expect("No elements in col") - sorted[0]) * var,
    )
}

struct Gauss {
    rng: std::cell::RefCell<StdRng>,
    mean: f64,
    var: f64,
}

impl Gauss {
    fn new(seed: u64, mean: f64, var: f64) -> Gauss {
        let rng = std::cell::RefCell::new(StdRng::seed_from_u64(seed));
        Gauss { rng, mean, var }
    }

    fn sample(&self) -> f64 {
        let mut a: f64 = self.rng.borrow_mut().random();
        // avoid 0
        while a.abs() < 1e-17 {
            a = self.rng.borrow_mut().random();
        }
        let b: f64 = self.rng.borrow_mut().random();
        // Box-Muller transform
        let z = f64::sqrt(-2.0 * a.ln()) * f64::cos(2.0 * std::f64::consts::PI * b);
        self.mean + self.var * z
    }
}

#[cfg(test)]
mod test {
    use crate::generators::mnar::transform;

    use super::*;

    #[test]
    fn create() {
        let _ = MNAR::new(None, None, None);
        let _ = MNAR::new(Some(0.5), Some(1.0), None);
    }

    #[test]
    fn _transform() {
        let v: &[f64] = &[1.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let (mean, var) = transform(v, 0.5, 1.0);
        assert!((mean - 2.5).abs() < 1e-17, "Actual Mean: {mean}");
        assert!((var - 4.0).abs() < 1e-17, "Actual Var: {var}");
        let v: &[f64] = &[0.0, 1.0, 2.0, 3.0, 4.0];
        let (mean, var) = transform(v, 0.25, 0.0);
        assert!((mean - 1.0).abs() < 1e-17, "Actual Mean: {mean}");
        assert!((var - 0.0).abs() < 1e-17, "Actual Var: {var}");
        let v: Vec<f64> = (0..10).map(|x| x as f64).collect();
        let (mean, var) = transform(&v, 0.5, 0.0);
        assert!((mean - 4.5).abs() < 1e-17, "Actual Mean: {mean}");
        assert!((var - 0.0).abs() < 1e-17, "Actual Var: {var}");
        let (mean, var) = transform(&v, 0.0, 1.0);
        assert!((mean - 0.0).abs() < 1e-17, "Actual Mean: {mean}");
        assert!((var - 9.0).abs() < 1e-17, "Actual Var: {var}");
        let (mean, var) = transform(&v, 1.0, 0.0);
        assert!((mean - 9.0).abs() < 1e-17, "Actual Mean: {mean}");
        assert!((var - 0.0).abs() < 1e-17, "Actual Var: {var}");
    }

    #[test]
    fn gaussian_stats() {
        let expected_mean = 2.0;
        let expected_stddev = 1.5;

        let g = Gauss::new(0, expected_mean, expected_stddev);

        let n = 100_000;

        let mut samples = Vec::with_capacity(n);

        for _ in 0..n {
            samples.push(g.sample());
        }
        let mean = samples.iter().sum::<f64>() / n as f64;
        let variance = samples
            .iter()
            .map(|x| {
                let d = x - mean;
                d * d
            })
            .sum::<f64>()
            / n as f64;

        let stddev = variance.sqrt();
        let tolerance = 0.15;
        assert!((mean - expected_mean).abs() < tolerance, "mean = {}", mean);
        assert!(
            (stddev - expected_stddev).abs() < tolerance,
            "stddev = {}",
            stddev
        );
    }
}
