//! The random draws the planner makes, named like Python's `random` module so the ported code reads the same.
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

pub struct Random(StdRng);

impl Random {
    pub fn seeded(seed: u64) -> Random {
        Random(StdRng::seed_from_u64(seed))
    }

    /// Uniform in [0, 1).
    pub fn random(&mut self) -> f64 {
        self.0.random::<f64>()
    }

    /// Uniform in [lo, hi].
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.random()
    }

    /// Integer in [lo, hi], both included.
    pub fn randint(&mut self, lo: usize, hi: usize) -> usize {
        self.0.random_range(lo..=hi)
    }

    /// Normal with mean `mu` and spread `sigma` (Box-Muller).
    pub fn gauss(&mut self, mu: f64, sigma: f64) -> f64 {
        let u1 = 1.0 - self.random(); // (0, 1], so the log is finite
        let u2 = self.random();
        mu + sigma * (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}
