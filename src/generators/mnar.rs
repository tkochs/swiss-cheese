use super::{
    common,
    common::Generator,
    common::mode::*,
    constants,
    utils::{Gauss, fix, get_distribution},
};
use ndarray::Array2;
use pyo3::prelude::*;
use rand::prelude::*;
use rayon::prelude::*;

#[pyclass(name = "MNAR")]
pub struct MNAR {
    mean: f64,
    variance: f64,
    block_size: Option<(f64, f64)>,
    max_missing_per_column: f64,
    mode: Mode,
    rng: StdRng,
}

#[pymethods]
impl MNAR {
    #[new]
    #[pyo3(signature = (mean=None, variance=None, block_size=None, max_missing_per_column=constants::MAX_MISSING_PER_COLUMN, mode="GM", random_seed=None))]
    fn new(
        mean: Option<f64>,
        variance: Option<f64>,
        block_size: Option<(f64, f64)>,
        max_missing_per_column: f64,
        mode: &str,

        random_seed: Option<u64>,
    ) -> PyResult<MNAR> {
        let mode: Mode = mode.try_into()?;
        mode.check_params(&mean, &variance, &block_size);
        let mean = mean.unwrap_or(constants::DEFAULT_MEAN);
        let variance = variance.unwrap_or(constants::DEFAULT_VAR);
        let rng = common::build_rng(random_seed);
        Ok(MNAR {
            mean,
            variance,
            block_size,
            max_missing_per_column,
            mode,
            rng,
        })
    }

    fn __call__<'py>(
        &mut self,
        py: Python<'py>,
        data: &Bound<'_, PyAny>,
        missing_rate: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.call(py, data, missing_rate)
    }

    fn __repr__(&self) -> String {
        format!("MNAR[{}]", self.mean)
    }
}
impl Generator for MNAR {
    fn max_missing_per_column(&self) -> f64 {
        self.max_missing_per_column
    }

    fn drop(&mut self, arr: &mut Array2<f64>, alpha: f64) {
        let n_missing = (arr.len() as f64 * alpha).ceil() as usize;
        let mut missing_count = 0;
        let distributions = get_distribution(self.mean, self.variance, arr.view());
        let cmp: Option<fn(&f64, &f64, &f64) -> std::cmp::Ordering> = match self.mode {
            Mode::MAX => Some(|a, b, _| a.total_cmp(b)),
            Mode::MIN => Some(|a, b, _| b.total_cmp(a)),
            Mode::GM => Some(|a, b, s| ((*a - s) * (*a - s)).total_cmp(&((*b - s) * (*b - s)))),
            Mode::BLOCK => None,
        };
        let fix = fix(arr.shape(), &mut self.rng);
        match self.mode {
            Mode::GM | Mode::MAX | Mode::MIN => {
                while missing_count < n_missing {
                    let cols = select_cols(&mut self.rng, arr, missing_count, n_missing);
                    missing_count += self.drop_cols(
                        arr,
                        &distributions,
                        &cols,
                        cmp.expect("Called wrong drop function!"),
                        &fix,
                    );
                }
            }
            Mode::BLOCK => {
                while missing_count < n_missing {
                    missing_count += self.drop_pattern(arr, &fix);
                }
            }
        }
    }
}

impl MNAR {
    fn drop_cols(
        &mut self,
        arr: &mut Array2<f64>,
        distributions: &[Gauss],
        cols: &[usize],
        cmp: fn(&f64, &f64, &f64) -> std::cmp::Ordering,
        fix: &[usize],
    ) -> usize {
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
                    .filter(|(i, v)| !v.is_nan() && fix[*i] != c)
                    .min_by(|(_, a), (_, b)| {
                        cmp(*a, *b, &s)
                        // ((*a - s) * (*a - s)).total_cmp(&((*b - s) * (*b - s)))
                    })
                    .map_or(None, |(i, _)| Some(i)); //&format!("No argmin found for {c}!"))
                (i, c)
            })
            .collect();
        let mut count = 0;
        for (opt, c) in indices {
            opt.map(|r| {
                arr[(r, c)] = f64::NAN;
                count += 1;
            });
        }
        count
    }

    fn drop_pattern(&mut self, arr: &mut Array2<f64>, fix: &[usize]) -> usize {
        let ids = match self.mode {
            Mode::BLOCK => {
                let (max_width, max_height) = self.block_size.unwrap();
                let (max_width, max_height) = (
                    (arr.ncols() as f64 * max_width) as usize,
                    (arr.nrows() as f64 * max_height) as usize,
                );
                let (x, y, widht, height): (usize, usize, usize, usize) = (
                    self.rng.random_range(0..arr.ncols()),
                    self.rng.random_range(0..arr.nrows()),
                    self.rng.random_range(0..max_width),
                    self.rng.random_range(0..max_height),
                );
                (0..widht).map(|i| (x + i, y))
            }
            _ => panic!("Not a pattern!"),
        };
        ids.for_each(|(x, y): (usize, usize)| {
            arr[(x, y)] = f64::NAN;
        });
        ids.len()
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
        let _ = MNAR::new(None, None, None, 0.8, "gm", None);
        let _ = MNAR::new(Some(0.5), Some(1.0), None, 0.8, "gm", None);
    }
}
