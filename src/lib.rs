use pyo3::prelude::*;
mod generators;
mod utils;

const MAX_AMOUNT_OF_WORK: u64 = 500_000;

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
