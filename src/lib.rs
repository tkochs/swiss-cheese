use pyo3::prelude::*;
mod generators;
mod utils;

#[pymodule]
mod swiss_cheese {
    use super::generators;
    use pyo3::prelude::*;

    #[pymodule_init]
    fn init(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add_class::<generators::MNAR>()?;
        module.add_class::<generators::MAR>()?;
        Ok(())
    }
}
