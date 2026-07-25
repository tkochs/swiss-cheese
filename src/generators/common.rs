use crate::utils::{StringEncoding, arr_to_out, pyany_to_vec};
use ndarray::Array2;
use pyo3::exceptions::PyUserWarning;
use pyo3::prelude::*;
use rand::prelude::*;
use std::sync::Arc;

pub mod mode {
    use super::*;
    pub enum Mode {
        GM,
        MAX,
        MIN,
        BLOCK,
    }
    pub enum Error {
        UnknownMode(String),
    }
    pub const ALLOWED_MODES: &[&str] = &["GM", "MAX", "MIN", "BLOCK"];

    impl TryInto<Mode> for &str {
        type Error = self::Error;

        fn try_into(self) -> Result<Mode, Self::Error> {
            match self.to_lowercase().as_str() {
                "gm" => Ok(Mode::GM),
                "max" => Ok(Mode::MAX),
                "min" => Ok(Mode::MIN),
                "block" => Ok(Mode::BLOCK),
                _ => Err(Error::UnknownMode(format!(
                    "Unknown mode parameter: {} (Alloed: {:?})",
                    self, ALLOWED_MODES
                ))),
            }
        }
    }
    impl From<Error> for pyo3::PyErr {
        fn from(err: Error) -> pyo3::PyErr {
            match err {
                Error::UnknownMode(s) => pyo3::exceptions::PyValueError::new_err(format!("{s}")),
            }
        }
    }

    impl Mode {
        pub fn check_params(&self, mean: &Option<f64>, variance: &Option<f64>) {
            if !matches!(self, Mode::GM) {
                if mean.is_none() || variance.is_none() {
                    pyo3::Python::attach(|py| {
                        PyErr::warn(
                            py,
                            &py.get_type::<PyUserWarning>(),
                            c"Mean/variance passed without passing a mode that supports it!",
                            0,
                        )
                        .expect("Something went wrong..");
                    });
                }
            }
        }
    }
}

#[inline]
pub fn _adjust_alpha(n_cols: usize, alpha: f64, max_missing_per_column: f64) -> f64 {
    let max = max_missing_per_column - (max_missing_per_column / n_cols as f64);
    if alpha > max {
        let msg = std::ffi::CString::new(format!(
            "Warning: Missing rate too high to ensure MAR properties! Maximum missing rate: {}",
            max
        ))
        .unwrap();
        pyo3::Python::attach(|py| {
            PyErr::warn(py, &py.get_type::<PyUserWarning>(), &msg, 0)
                .expect("Something went wrong..");
        });
        max
    } else {
        alpha
    }
}

pub fn build_rng(seed: Option<u64>) -> StdRng {
    let seed = seed.unwrap_or_else(|| rand::rng().random());
    StdRng::seed_from_u64(seed)
}

pub trait Generator {
    fn call<'py>(
        &mut self,
        py: Python<'py>,
        data: &Bound<'_, PyAny>,
        missing_rate: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (array, out, enc_info) = pyany_to_vec(data, &Some(StringEncoding::LabelEncoding))?;
        let mut arr = Arc::new(array);
        let missing_rate = _adjust_alpha(arr.ncols(), missing_rate, self.max_missing_per_column());
        self.drop(&mut arr, missing_rate);
        arr_to_out(py, &arr, out, enc_info)
    }
    fn max_missing_per_column(&self) -> f64;
    fn drop(&mut self, arr: &mut Arc<Array2<f64>>, alpha: f64);
}
