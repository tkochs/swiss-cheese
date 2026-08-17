use pyo3::prelude::*;
use std::fmt;

#[derive(Debug)]
pub enum Errors {
    ValueError(String),
    MaxWorkExeeded,
}
impl fmt::Display for Errors {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValueError(msg) => write!(f, "{}", msg),
            MaxWorkExeeded => write!(
                f,
                "After {} iterations the desired amount of missing_rate has not been reached stopping now!",
                MAX_AMOUNT_OF_WORK
            ),
        }
    }
}
impl std::error::Error for Errors {}
use pyo3::exceptions::{PyRuntimeError, PyValueError};

use crate::{
    MAX_AMOUNT_OF_WORK,
    utils::Errors::{MaxWorkExeeded, ValueError},
};

impl From<Errors> for PyErr {
    fn from(err: Errors) -> PyErr {
        match err {
            Errors::ValueError(_) => PyValueError::new_err(err.to_string()),
            _ => PyRuntimeError::new_err(err.to_string()),
        }
    }
}
