use pyo3::prelude::*;
mod generators;
mod utils;

/// A Python module implemented in Rust.
#[pymodule]
mod swiss_cheese {
    use super::generators;
    use pyo3::prelude::*;

    #[pymodule_init]
    fn init(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add_class::<generators::MNAR>()?;
        Ok(())
    }
}
