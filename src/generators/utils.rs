use rand::prelude::*;

pub struct Gauss {
    mean: f64,
    var: f64,
}

impl Gauss {
    pub fn new(mean: f64, var: f64) -> Gauss {
        // let rng = std::cell::RefCell::new(StdRng::seed_from_u64(seed));
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
pub fn fix() {
    // "TODO: implement";
    // Fixes on value per row, that way every datapoint has at least one observed feature
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
}
