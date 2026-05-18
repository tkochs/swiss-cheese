use ndarray::Array2;
use numpy::{IntoPyArray, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

pub enum OUT {
    Numpy,
    DataFrame(Vec<String>),
}

const SUPPORTED_TYPES: &str = "numpy.ndarray, pandas.DataFrame";

pub fn pyany_to_vec(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
) -> PyResult<((Vec<f64>, usize, usize), OUT)> {
    // 1. numpy — try fast path first, fall back to buffer protocol
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<f64>>() {
        let shape = arr.shape();
        let (nrows, ncols) = (shape[0], shape[1]);
        let data = match arr.as_slice() {
            Ok(s) => s.to_vec(),
            Err(_) => arr.as_array().iter().copied().collect(),
        };
        return Ok(((data, nrows, ncols), OUT::Numpy));
    }
    // buffer-protocol fallback for numpy (handles stale type cache)
    let type_name = obj
        .get_type()
        .qualname()
        .map(|s| s.to_string())
        .unwrap_or_default();
    if type_name == "ndarray" {
        let shape: Vec<usize> = obj.getattr("shape")?.extract()?;
        if shape.len() == 2 {
            let (nrows, ncols) = (shape[0], shape[1]);
            let data = read_via_buffer(py, obj, nrows, ncols)?;
            return Ok(((data, nrows, ncols), OUT::Numpy));
        }
    }

    // 2. pandas — never use isinstance on the derived numpy array
    let pandas = py.import("pandas")?;
    if obj.is_instance(&pandas.getattr("DataFrame")?)? {
        let shape: Vec<usize> = obj.getattr("shape")?.extract()?;
        let (nrows, ncols) = (shape[0], shape[1]);
        let columns: Vec<String> = obj
            .getattr("columns")?
            .try_iter()?
            .map(|item| {
                item.expect("no items found!")
                    .str()
                    .expect("Failed str conversion")
                    .to_string()
            })
            .collect();

        let kwargs = PyDict::new(py);
        kwargs.set_item("dtype", "float64")?;
        kwargs.set_item("copy", false)?;
        let np_any = obj.call_method("to_numpy", (), Some(&kwargs))?;
        let data = read_via_buffer(py, &np_any, nrows, ncols)?;

        return Ok(((data, nrows, ncols), OUT::DataFrame(columns)));
    }

    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
        "Unsupported type: '{}'. Supported types are: {}",
        type_name, SUPPORTED_TYPES
    )))
}

fn read_via_buffer(
    py: Python<'_>,
    arr: &Bound<'_, PyAny>,
    nrows: usize,
    ncols: usize,
) -> PyResult<Vec<f64>> {
    let buf = PyBuffer::<f64>::get(arr)?;
    if buf.dimensions() != 2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Expected a 2-dimensional array",
        ));
    }
    let mut data = vec![0f64; nrows * ncols];
    buf.copy_to_slice(py, &mut data)?;
    Ok(data)
}

pub fn arr_to_out<'py>(
    py: Python<'py>,
    arr: &Array2<f64>,
    out: OUT,
) -> PyResult<Bound<'py, PyAny>> {
    match out {
        OUT::Numpy => Ok(arr.to_owned().into_pyarray(py).into_any()),
        OUT::DataFrame(columns) => {
            let pd = PyModule::import(py, "pandas")?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("columns", columns)?;
            let df = pd
                .getattr("DataFrame")?
                .call((arr.to_owned().into_pyarray(py),), Some(&kwargs))?;
            Ok(df.into_any())
        }
    }
}
