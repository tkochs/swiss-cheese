use ndarray::ArrayView2;
use rand::prelude::*;
use rayon::prelude::*;

pub struct Gauss {
    mean: f64,
    var: f64,
}

impl Gauss {
    pub fn new(mean: f64, var: f64) -> Gauss {
        Gauss { mean, var }
    }

    pub fn sample(&self, rng: &mut StdRng) -> f64 {
        let mut a: f64 = rng.random();
        // avoid 0
        while a.abs() < 1e-17 {
            a = rng.random();
        }
        let b: f64 = rng.random();
        // Box-Muller transform
        let z = f64::sqrt(-2.0 * a.ln()) * f64::cos(2.0 * std::f64::consts::PI * b);
        self.mean + self.var * z
    }
}

pub fn get_distribution(mean: f64, var: f64, arr: ArrayView2<f64>) -> Vec<Gauss> {
    (0..arr.ncols())
        .into_par_iter()
        .map(|i| {
            let col = arr.column(i);
            let mut buff = Vec::with_capacity(arr.nrows());
            let (local_mean, local_var) = transform(&mut buff, col.iter().copied(), mean, var);
            Gauss::new(local_mean, local_var)
        })
        .collect()
}

fn transform(
    buff: &mut Vec<f64>,
    col: impl Iterator<Item = f64>,
    mean: f64,
    var: f64,
) -> (f64, f64) {
    buff.clear();
    buff.extend(col);
    let n = (buff.len() as f64 * mean).floor() as usize;
    buff.sort_unstable_by(|a, b| a.total_cmp(b));
    let q = if buff.len() % 2 == 1 || n == 0 {
        buff[n]
    } else {
        (buff[n - 1] + buff.get(n).unwrap_or_else(|| &buff[n - 1])) / 2.0
    };
    (
        q,
        (buff.last().expect("No elements in col") - buff[0]) * var,
    )
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn gaussian_stats() {
        let mut rng = StdRng::seed_from_u64(43);
        let expected_mean = 2.0;
        let expected_stddev = 1.5;

        let g = Gauss::new(expected_mean, expected_stddev);

        let n = 100_000;

        let mut samples = Vec::with_capacity(n);

        for _ in 0..n {
            samples.push(g.sample(&mut rng));
        }
        let mean = samples.iter().sum::<f64>() / n as f64;
        let variance = samples
            .iter()
            .map(|x| {
                let d = x - mean;
                d * d
            })
            .sum::<f64>()
            / n as f64;

        let stddev = variance.sqrt();
        let tolerance = 0.15;
        assert!((mean - expected_mean).abs() < tolerance, "mean = {}", mean);
        assert!(
            (stddev - expected_stddev).abs() < tolerance,
            "stddev = {}",
            stddev
        );
    }

    #[test]
    fn _transform() {
        let v = [1.0, 1.0, 2.0, 3.0, 4.0, 5.0].iter().copied();
        let mut buff: Vec<f64> = vec![0.0; v.len()];
        let (mean, var) = transform(&mut buff, v, 0.5, 1.0);
        assert!((mean - 2.5).abs() < 1e-17, "Actual Mean: {mean}");
        assert!((var - 4.0).abs() < 1e-17, "Actual Var: {var}");
        let v = [0.0, 1.0, 2.0, 3.0, 4.0].iter().copied();
        let mut buff: Vec<f64> = vec![0.0; v.len()];
        let (mean, var) = transform(&mut buff, v, 0.25, 0.0);
        assert!((mean - 1.0).abs() < 1e-17, "Actual Mean: {mean}");
        assert!((var - 0.0).abs() < 1e-17, "Actual Var: {var}");
        let v: Vec<f64> = (0..10).map(|x| x as f64).collect();
        let mut buff: Vec<f64> = vec![0.0; v.len()];
        let (mean, var) = transform(&mut buff, v.iter().copied(), 0.5, 0.0);
        assert!((mean - 4.5).abs() < 1e-17, "Actual Mean: {mean}");
        assert!((var - 0.0).abs() < 1e-17, "Actual Var: {var}");
        let (mean, var) = transform(&mut buff, v.iter().copied(), 0.0, 1.0);
        assert!((mean - 0.0).abs() < 1e-17, "Actual Mean: {mean}");
        assert!((var - 9.0).abs() < 1e-17, "Actual Var: {var}");
        let (mean, var) = transform(&mut buff, v.iter().copied(), 1.0, 0.0);
        assert!((mean - 9.0).abs() < 1e-17, "Actual Mean: {mean}");
        assert!((var - 0.0).abs() < 1e-17, "Actual Var: {var}");
    }
}
