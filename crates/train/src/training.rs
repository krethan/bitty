//! Stochastic-flip training primitives.
//!
//! Moved out of `bitllm-hip-tern` (`mercurial.rs`) when the 1-bit training
//! path landed. `TrainingConfig` drives the annealing schedule and
//! `StochasticFlip` perturbs accumulated gradients with gaussian noise whose
//! scale anneals to zero; `TernaryLoRA` (see [`crate::lora`]) uses the same
//! schedule to decide discrete trit flips.

/// Training configuration. Defaults keep the historical values (learning
/// rate, Adam-style moments, weight decay, annealing) that `mercurial.rs`
/// used when this type lived in `bitllm-hip-tern`.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub weight_decay: f32,
    pub epsilon: f32,
    pub grad_clip: f32,
    pub noise_scale: f32,
    pub noise_anneal_steps: u64,
    /// Flip dead-zone in trit units: a gradient vote must exceed this (after
    /// normalization by the effective scale product) before a trit moves.
    pub flip_threshold: f32,
    /// RNG seed for reproducible noise.
    pub seed: u64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1.5e-4,
            beta1: 0.9,
            beta2: 0.98,
            weight_decay: 0.01,
            epsilon: 1e-8,
            grad_clip: 1.0,
            noise_scale: 0.05,
            noise_anneal_steps: 1_000_000_000_000, // 1 trillion tokens
            flip_threshold: 0.5,
            seed: 42,
        }
    }
}

/// Deterministic xorshift64 RNG with gaussian draws (Box-Muller). Used instead
/// of the global `rand::random` so training noise is reproducible.
pub(crate) struct SeededRng {
    state: u64,
}

impl SeededRng {
    pub(crate) fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        self.state.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// Uniform in `[0.0, 1.0)`.
    pub(crate) fn next_unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard gaussian via Box-Muller.
    pub(crate) fn next_gaussian(&mut self) -> f32 {
        let u1 = self.next_unit().max(1e-10);
        let u2 = self.next_unit();
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }
}

/// Shared half-cosine annealing: `noise_scale` at step 0 decaying to ~0 at
/// `noise_anneal_steps`.
pub fn annealed_noise_scale(step: u64, noise_scale: f32, noise_anneal_steps: u64) -> f32 {
    if noise_anneal_steps == 0 {
        return noise_scale;
    }
    let progress = ((step as f64) / (noise_anneal_steps as f64)).min(1.0);
    let half_cos = 0.5 * (1.0 + (progress * std::f64::consts::PI).cos());
    noise_scale * half_cos as f32
}

/// Stochastic Flip Regularizer.
///
/// Maintains a per-parameter EMA of gradient variance and perturbs the
/// incoming gradients with gaussian noise whose standard deviation anneals to
/// zero. The noisy, clipped gradients are the "votes" that discrete training
/// paths (e.g. [`crate::lora::TernaryLoRA`]) threshold to flip trits.
pub struct StochasticFlip {
    config: TrainingConfig,
    step: u64,
    grad_var_ema: Vec<f32>,
    rng: SeededRng,
}

impl StochasticFlip {
    pub fn new(config: TrainingConfig, num_params: usize) -> Self {
        Self {
            rng: SeededRng::new(config.seed),
            config,
            step: 0,
            grad_var_ema: vec![0.0; num_params],
        }
    }

    /// Compute the current noise scale based on annealing schedule.
    pub fn current_noise_scale(&self) -> f32 {
        annealed_noise_scale(
            self.step,
            self.config.noise_scale,
            self.config.noise_anneal_steps,
        )
    }

    /// Draw one gaussian noise sample at the current annealed scale.
    pub fn noise_draw(&mut self) -> f32 {
        self.rng.next_gaussian() * self.current_noise_scale()
    }

    /// Apply stochastic flip to gradients.
    pub fn apply(&mut self, grads: &mut [f32]) {
        let noise_scale = self.current_noise_scale();

        for (i, g) in grads.iter_mut().enumerate() {
            // Update EMA of gradient variance.
            self.grad_var_ema[i] = 0.99 * self.grad_var_ema[i] + 0.01 * g.powi(2);

            // Compute noise.
            let noise_std = noise_scale * self.grad_var_ema[i].sqrt();
            let noise = self.rng.next_gaussian() * noise_std;

            // Apply noise and clip.
            *g += noise;
            *g = g.clamp(-2.0 * noise_std, 2.0 * noise_std);
        }

        self.step += 1;
    }

    pub fn step(&self) -> u64 {
        self.step
    }

    /// Advance the annealing step counter without perturbing gradients (used
    /// by discrete trainers like [`crate::lora::TernaryLoRA`] that draw from
    /// the noise schedule directly rather than through [`Self::apply`]).
    pub fn advance(&mut self) {
        self.step += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_config_defaults() {
        let config = TrainingConfig::default();
        assert_eq!(config.learning_rate, 1.5e-4);
        assert_eq!(config.beta1, 0.9);
        assert_eq!(config.beta2, 0.98);
        assert_eq!(config.noise_scale, 0.05);
        assert_eq!(config.flip_threshold, 0.5);
        assert_eq!(config.seed, 42);
    }

    #[test]
    fn noise_anneals_to_zero() {
        let config = TrainingConfig {
            noise_anneal_steps: 100,
            ..TrainingConfig::default()
        };
        let flip = StochasticFlip::new(config, 4);
        let start = flip.current_noise_scale();
        let mut far = flip;
        far.step = 100;
        let end = far.current_noise_scale();
        assert!(start > 0.0);
        assert!(end.abs() < 1e-3, "noise must anneal to ~0, got {end}");
    }

    #[test]
    fn deterministic_with_seed() {
        let config = TrainingConfig::default();
        let mut a = StochasticFlip::new(config.clone(), 8);
        let mut b = StochasticFlip::new(config, 8);
        let mut ga = vec![0.3f32; 8];
        let mut gb = ga.clone();
        a.apply(&mut ga);
        b.apply(&mut gb);
        assert_eq!(ga, gb, "same seed must reproduce the same noise");
    }

    #[test]
    fn apply_updates_variance_ema_and_step() {
        let config = TrainingConfig {
            noise_anneal_steps: 1_000_000,
            ..TrainingConfig::default()
        };
        let mut flip = StochasticFlip::new(config, 2);
        let mut grads = vec![0.5f32, -0.25];
        flip.apply(&mut grads);
        assert_eq!(flip.step(), 1);
        assert!(flip.grad_var_ema[0] > 0.0);
        // Noise is annealed-down and clipped, but the gradient sign survives.
        assert!(grads[0] > 0.0);
        assert!(grads[1] < 0.0);
    }

    #[test]
    fn gaussian_draws_are_finite() {
        let config = TrainingConfig::default();
        let mut flip = StochasticFlip::new(config, 1);
        for _ in 0..1000 {
            let d = flip.noise_draw();
            assert!(d.is_finite());
        }
    }
}
