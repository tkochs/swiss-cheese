use crate::{
    generators::common::mode::Mode,
    generators::constants,
    utils::{SendPtr, StringEncoding, arr_to_out, pyany_to_vec},
};
use ndarray::Array2;
use pyo3::exceptions::PyUserWarning;
use pyo3::prelude::*;
use rand::prelude::*;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::sync::Arc;

pub mod mode {
    use super::*;
    pub enum Mode {
        GM(f64, f64),
        MAX,
        MIN,
        BLOCK((f64, f64)),
        BLOB(usize),
    }
    pub enum Error {
        UnknownMode(String),
    }
    pub const ALLOWED_MODES: &[&str] = &["GM", "MAX", "MIN", "BLOCK", "BLOB"];

    impl Into<&str> for &Mode {
        fn into(self) -> &'static str {
            match self {
                Mode::GM(_, _) => "GM",
                Mode::MAX => "MAX",
                Mode::MIN => "MIN",
                Mode::BLOCK(_) => "BLOCK",
                Mode::BLOB(_) => "BLOB",
            }
        }
    }

    // impl TryInto<Mode> for &str {
    //     type Error = self::Error;
    //
    //     fn try_into(self) -> Result<Mode, Self::Error> {
    //         match self.to_lowercase().as_str() {
    //             "gm" => Ok(Mode::GM),
    //             "max" => Ok(Mode::MAX),
    //             "min" => Ok(Mode::MIN),
    //             "block" => Ok(Mode::BLOCK),
    //             "blob" => Ok(Mode::BLOB),
    //             _ => Err(Error::UnknownMode(format!(
    //                 "Unknown mode parameter: {} (Alloed: {:?})",
    //                 self, ALLOWED_MODES
    //             ))),
    //         }
    //     }
    // }
    impl From<Error> for pyo3::PyErr {
        fn from(err: Error) -> pyo3::PyErr {
            match err {
                Error::UnknownMode(s) => pyo3::exceptions::PyValueError::new_err(format!("{s}")),
            }
        }
    }

    impl Mode {
        pub fn new(
            value: &str,
            mean: Option<f64>,
            variance: Option<f64>,
            block_size: Option<(f64, f64)>,
            n_blobs: Option<usize>,
        ) -> Result<Self, Error> {
            let mode = match value.to_lowercase().as_str() {
                "gm" => Mode::GM(
                    mean.unwrap_or(constants::DEFAULT_MEAN),
                    variance.unwrap_or(constants::DEFAULT_VAR),
                ),
                "max" => Mode::MAX,
                "min" => Mode::MIN,
                "block" => Mode::BLOCK(block_size.unwrap_or(constants::DEFAULT_BLOCK_SIZE)),
                "blob" => Mode::BLOB(n_blobs.unwrap_or(constants::DEFAULT_BLOBS)),
                _ => {
                    return Err(Error::UnknownMode(format!(
                        "Unknown mode parameter: {} (Alloed: {:?})",
                        value, ALLOWED_MODES
                    )));
                }
            };
            mode.check_params(&mean, &variance, &block_size, &n_blobs);
            Ok(mode)
        }
        pub fn check_params(
            &self,
            mean: &Option<f64>,
            variance: &Option<f64>,
            block_size: &Option<(f64, f64)>,
            n_blobs: &Option<usize>,
        ) {
            if unsafe { pyo3::ffi::Py_IsInitialized() } == 0 {
                // only throw pyton warnings if Python if called from python
                return;
            }
            if !matches!(self, Mode::GM(_, _)) {
                if mean.is_some() || variance.is_some() {
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
            if !matches!(self, Mode::BLOCK(_)) {
                if block_size.is_some() {
                    pyo3::Python::attach(|py| {
                        PyErr::warn(
                            py,
                            &py.get_type::<PyUserWarning>(),
                            c"Block size passed without passing a mode that supports it!",
                            0,
                        )
                        .expect("Something went wrong..");
                    });
                }
            }
            if !matches!(self, Mode::BLOB(_)) {
                if n_blobs.is_some() {
                    pyo3::Python::attach(|py| {
                        PyErr::warn(
                            py,
                            &py.get_type::<PyUserWarning>(),
                            c"Number of Blobs passed without passing a mode that supports it!",
                            0,
                        )
                        .expect("Something went wrong..");
                    });
                }
            }
        }

        pub fn as_str(&self) -> &str {
            self.into()
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
        let (mut arr, out, enc_info) = pyany_to_vec(data, &Some(StringEncoding::LabelEncoding))?;
        let missing_rate = _adjust_alpha(arr.ncols(), missing_rate, self.max_missing_per_column());
        self.drop(&mut arr, missing_rate);
        arr_to_out(py, &arr, out, enc_info)
    }
    fn max_missing_per_column(&self) -> f64;
    fn drop(&mut self, arr: &mut Array2<f64>, alpha: f64);
}

pub fn remove(arr: &mut Array2<f64>, ids: &Vec<(usize, usize)>) -> usize {
    let arr_ptr = Arc::new(SendPtr(arr.as_mut_ptr()));
    println!("{:?}", &ids);
    let l = ids.len();
    ids.par_iter().for_each(|(r, c)| {
        // arr[(x, y)] = f64::NAN;
        unsafe {
            *arr_ptr.0.add(r * arr.ncols() + c) = f64::NAN;
        }
    });
    l
}

pub fn fix(shape: &[usize], rng: &mut StdRng, mode: &Mode) -> Vec<usize> {
    let &[rows, cols, ..] = shape else {
        panic!("fix() needs at least [rows, cols]");
    };
    match mode {
        Mode::BLOCK(_) => {
            let col = rng.random_range(0..cols);
            (0..rows).into_iter().map(|_| col).collect()
        }
        _ => (0..rows)
            .into_iter()
            .map(|_| rng.random_range(0..cols))
            .collect(),
    }
}
