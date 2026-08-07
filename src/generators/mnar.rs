use core::f64;
use std::collections::HashSet;

use super::{
    common,
    common::Generator,
    common::mode::*,
    constants,
    utils::{Gauss, get_distribution},
};
use crate::generators::utils::get_samples;
use ndarray::Array2;
use pyo3::prelude::*;
use rand::prelude::*;
use rayon::prelude::*;

#[pyclass(name = "MNAR")]
pub struct MNAR {
    max_missing_per_column: f64,
    mode: Mode,
    rng: StdRng,
}

#[pymethods]
impl MNAR {
    #[new]
    #[pyo3(signature = (mean=None, variance=None, block_size=None, n_blobs=None, max_missing_per_column=constants::MAX_MISSING_PER_COLUMN, mode="GM", random_seed=None))]
    fn new(
        mean: Option<f64>,
        variance: Option<f64>,
        block_size: Option<(f64, f64)>,
        n_blobs: Option<usize>,
        max_missing_per_column: f64,
        mode: &str,

        random_seed: Option<u64>,
    ) -> PyResult<MNAR> {
        let mode: Mode = Mode::new(mode, mean, variance, block_size, n_blobs)?;
        let rng = common::build_rng(random_seed);
        Ok(MNAR {
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
        match self.mode {
            Mode::GM { mean, .. } => format!("MNAR[{}]", mean),
            Mode::MIN => format!("MNAR[MIN]"),
            Mode::MAX => format!("MNAR[MAX]"),
            Mode::BLOCK {
                max_width,
                max_height,
            } => format!("MNAR[Blocks({}, {})]", max_width, max_height),
            Mode::BLOB(n) => format!("MNAR[Blobs({})]", n),
        }
    }
}
impl Generator for MNAR {
    fn max_missing_per_column(&self) -> f64 {
        self.max_missing_per_column
    }

    fn drop(&mut self, arr: &mut Array2<f64>, alpha: f64) {
        let n_missing = (arr.len() as f64 * alpha).ceil() as usize;
        let mut missing_count = 0;
        let (distributions, cmp): (
            Option<Vec<Gauss>>,
            Option<fn(&f64, &f64, &f64) -> std::cmp::Ordering>,
        ) = match self.mode {
            Mode::MAX => (None, Some(|a, b, _| a.total_cmp(b))),
            Mode::MIN => (None, Some(|a, b, _| b.total_cmp(a))),
            Mode::GM { mean, var } => (
                Some(get_distribution(mean, var, arr.view())),
                Some(|a, b, s| ((*a - s) * (*a - s)).total_cmp(&((*b - s) * (*b - s)))),
            ),
            Mode::BLOCK { .. } => (None, None),
            Mode::BLOB(_) => (None, None),
        };
        let fix = common::fix(arr.shape(), &mut self.rng, &self.mode);
        match self.mode {
            Mode::GM { .. } | Mode::MAX | Mode::MIN => {
                let cmp = cmp.expect(&format!(
                    "Cmp function required for this mode: [{}]",
                    self.mode.as_str()
                ));
                while missing_count < n_missing {
                    let cols = select_cols(&mut self.rng, arr, missing_count, n_missing);
                    let samples = get_samples(&distributions, &cols, &mut self.rng);
                    missing_count += self.drop_cols(arr, &samples, &cols, cmp, &fix);
                }
            }
            Mode::BLOCK { .. } | Mode::BLOB(_) => {
                while missing_count < n_missing {
                    missing_count += self.drop_pattern(arr, &fix, n_missing - missing_count);
                }
            }
        }
    }
}

impl MNAR {
    fn drop_cols(
        &mut self,
        arr: &mut Array2<f64>,
        samples: &[f64],
        cols: &[usize],
        cmp: fn(&f64, &f64, &f64) -> std::cmp::Ordering,
        fix: &[usize],
    ) -> usize {
        let indices: Vec<_> = cols
            .par_iter()
            .zip(samples)
            .filter_map(|(&c, s)| {
                assert!(!s.is_nan(), "Sample is nan");
                arr.column(c)
                    .iter()
                    .enumerate()
                    .filter(|(i, v)| !v.is_nan() && unsafe { *fix.get_unchecked(*i) != c })
                    .min_by(|(_, a), (_, b)| {
                        cmp(*a, *b, s)
                        // ((*a - s) * (*a - s)).total_cmp(&((*b - s) * (*b - s)))
                    })
                    .map(|(i, _)| (i, c)) //&format!("No argmin found for {c}!"))
            })
            .collect();
        common::remove(arr, &indices)
    }

    fn drop_pattern(&mut self, arr: &mut Array2<f64>, fix: &[usize], nmiss: usize) -> usize {
        let ids: Vec<_> = match self.mode {
            Mode::BLOCK {
                max_width,
                max_height,
            } => {
                let (mut max_width, mut max_height) = (
                    (arr.ncols() as f64 * max_width) as usize,
                    (arr.nrows() as f64 * max_height) as usize - 1,
                );
                let (mut x, mut y, mut width, mut height);
                let mut ids;
                loop {
                    x = self.rng.random_range(0..arr.ncols());
                    y = self.rng.random_range(0..arr.nrows());
                    width = self
                        .rng
                        .random_range(1..=max_width.min(arr.ncols() - x).max(1));
                    height = self
                        .rng
                        .random_range(1..=max_height.min(arr.nrows() - y).max(1));
                    let mut intersect = false;
                    fix.iter().enumerate().for_each(|(row, &column)| {
                        if (column >= x && column < x + width) && (row >= y && row < y + height) {
                            intersect = true;
                        }
                    });

                    if width * height > nmiss {
                        continue;
                    }
                    ids = (0..width * height)
                        .map(|i| {
                            let (r, c) = (y + i / width, x + i % width);
                            if unsafe { arr.uget((r, c)).is_nan() } {
                                intersect = true;
                            }
                            (r, c)
                        })
                        .collect();
                    if !intersect {
                        break;
                    }
                    max_width -= 1;
                    max_height -= 1;
                }
                ids
            }
            Mode::BLOB(n) => {
                let mut centers = Vec::with_capacity(n);
                let mut n_per_center = Vec::with_capacity(n);
                let mut seeds: Vec<u64> = Vec::with_capacity(n);
                let mut nmiss = nmiss;
                for i in 0..n {
                    centers.push((
                        self.rng.random_range(0..arr.ncols()),
                        self.rng.random_range(0..arr.nrows()),
                    ));
                    n_per_center.push(if i != n - 1 {
                        self.rng.random_range(0..=nmiss.max(1))
                    } else {
                        nmiss
                    });
                    nmiss = nmiss.saturating_sub(n_per_center[i]);
                    seeds.push(self.rng.random());
                }

                centers
                    .par_iter()
                    .zip(&n_per_center)
                    .zip(&seeds)
                    .map(|((&(x, y), &k), &s)| {
                        let (vx, vy) = constants::DEFAULT_BLOB_VAR;
                        let dx = Gauss::new(x as f64, vx * arr.ncols() as f64);
                        let dy = Gauss::new(y as f64, vy * arr.nrows() as f64);
                        let mut rng = StdRng::seed_from_u64(s);
                        let mut points = HashSet::with_capacity(k);
                        for _ in 0..k {
                            'search: loop {
                                let a = (dx.sample(&mut rng) as usize).min(arr.ncols() - 1);
                                let b = (dy.sample(&mut rng) as usize).min(arr.nrows() - 1);
                                let p = (b, a);
                                if fix[b] == a || arr[p].is_nan() {
                                    continue;
                                }
                                if points.insert(p) {
                                    break;
                                } else {
                                    for radius in 0..arr.ncols() {
                                        let rmin = b.saturating_sub(radius);
                                        let rmax = (b + radius).min(arr.nrows() - 1);
                                        let cmin = a.saturating_sub(radius);
                                        let cmax = (a + radius).min(arr.ncols() - 1);

                                        // Search the perimeter of the square
                                        for r in rmin..=rmax {
                                            for c in cmin..=cmax {
                                                if (r > rmin && r < rmax && c > cmin && c < cmax)
                                                    || fix[r] == c
                                                {
                                                    continue;
                                                }

                                                if points.insert((r, c)) {
                                                    break 'search;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        points
                    })
                    .flatten()
                    .collect()
            }
            _ => panic!("Not a pattern!"),
        };
        common::remove(arr, &ids)
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
        let _ = MNAR::new(None, None, None, None, 0.8, "gm", None);
        let _ = MNAR::new(Some(0.5), Some(1.0), None, None, 0.8, "gm", None);
    }
}
