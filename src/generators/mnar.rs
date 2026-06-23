use super::{
    common::mode::*,
    constants,
    utils::{fix, get_distribution, Gauss},
};
use crate::utils::{arr_to_out, pyany_to_vec, StringEncoding};
use ndarray::Array2;
use pyo3::prelude::*;
use rand::prelude::*;
use rayon::prelude::*;
use std::sync::Arc;

#[pyclass(name = "MNARrs")]
pub struct MNAR {
    mean: f64,
    variance: f64,
    mode: Mode,
    rng: StdRng,
}

#[pymethods]
impl MNAR {
    #[new]
    #[pyo3(signature = (mean= None, variance=None, mode="GM", random_seed=None))]
    fn new(mean: Option<f64>, variance: Option<f64>, mode: &str, random_seed: Option<u64>) -> MNAR {
        let mean = mean.unwrap_or(constants::DEFAULT_MEAN);
        let variance = variance.unwrap_or(constants::DEFAULT_VAR);
        let seed = random_seed.unwrap_or(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("Getting time failed")
                .as_nanos() as u64,
        );

        let rng = StdRng::seed_from_u64(seed);
        MNAR {
            mean,
            variance,
            mode: mode.into(),
            rng,
        }
    }

    fn __call__<'py>(
        &mut self,
        py: Python<'py>,
        data: &Bound<'_, PyAny>,
        alpha: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (array, out, enc_info) = pyany_to_vec(data, &Some(StringEncoding::LabelEncoding))?;
        fix();
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
        let distributions = get_distribution(self.mean, self.variance, arr.view());
        let cmp: fn(&f64, &f64, &f64) -> std::cmp::Ordering = match self.mode {
            Mode::MAX => |a, b, _| a.total_cmp(b),
            Mode::MIN => |a, b, _| b.total_cmp(a),
            Mode::GM => |a, b, s| ((*a - s) * (*a - s)).total_cmp(&((*b - s) * (*b - s))),
        };
        while missing_count < n_missing {
            let cols = select_cols(&mut self.rng, arr, missing_count, n_missing);
            // self.drop_cols(arr, &distributions, &cols);
            self.drop_cols(arr, &distributions, &cols, cmp);
            missing_count += cols.len();
        }
    }

    fn drop_cols(
        &mut self,
        arr: &mut Arc<Array2<f64>>,
        distributions: &[Gauss],
        cols: &[usize],
        cmp: fn(&f64, &f64, &f64) -> std::cmp::Ordering,
    ) {
        let samples: Vec<f64> = cols
            .iter()
            .map(|&c| distributions[c].sample(&mut self.rng))
            .collect();
        let indices: Vec<_> = cols
            .par_iter()
            .zip(&samples)
            .map(|(&c, &s)| {
                assert!(!s.is_nan(), "Sample is nan");
                let i = arr
                    .column(c)
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| !v.is_nan())
                    .min_by(|(_, a), (_, b)| {
                        cmp(*a, *b, &s)
                        // ((*a - s) * (*a - s)).total_cmp(&((*b - s) * (*b - s)))
                    })
                    .expect("No argmin found!")
                    .0;
                (i, c)
            })
            .collect();
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

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn create() {
        let _ = MNAR::new(None, None, "gm", None);
        let _ = MNAR::new(Some(0.5), Some(1.0), "gm", None);
    }
}
