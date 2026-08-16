use bitllm_tensor::simd;
use rand::Rng;

#[derive(Debug, Clone)]
pub enum SamplingStrategy {
    Greedy,
    Temperature { temperature: f32 },
    TopK { k: usize, temperature: f32 },
    TopP { p: f32, temperature: f32 },
}

#[derive(Debug, Clone)]
pub struct Sampler {
    strategy: SamplingStrategy,
}

impl Sampler {
    pub fn new(strategy: SamplingStrategy) -> Self {
        Self { strategy }
    }

    pub fn greedy() -> Self {
        Self::new(SamplingStrategy::Greedy)
    }

    pub fn temperature(temp: f32) -> Self {
        Self::new(SamplingStrategy::Temperature { temperature: temp })
    }

    pub fn top_k(k: usize, temp: f32) -> Self {
        Self::new(SamplingStrategy::TopK {
            k,
            temperature: temp,
        })
    }

    pub fn sample(&self, logits: &[f32]) -> u32 {
        match &self.strategy {
            SamplingStrategy::Greedy => greedy_sample(logits),
            SamplingStrategy::Temperature { temperature } => temp_sample(logits, *temperature),
            SamplingStrategy::TopK { k, temperature } => top_k_sample(logits, *k, *temperature),
            SamplingStrategy::TopP { p, temperature } => top_p_sample(logits, *p, *temperature),
        }
    }
}

fn greedy_sample(logits: &[f32]) -> u32 {
    let max_val = simd::f32_max(logits);
    logits.iter().position(|&v| v == max_val).unwrap_or(0) as u32
}

fn temp_sample(logits: &[f32], temperature: f32) -> u32 {
    if temperature <= 0.0 {
        return greedy_sample(logits);
    }

    let probs = softmax(logits, temperature);
    sample_from_probs(&probs)
}

fn top_k_sample(logits: &[f32], k: usize, temperature: f32) -> u32 {
    let k = k.min(logits.len());
    if k == 0 {
        return 0;
    }
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.select_nth_unstable_by(k - 1, |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    indexed.truncate(k);

    let max_val = indexed
        .iter()
        .map(|&(_, v)| v)
        .fold(f32::NEG_INFINITY, f32::max);
    let inv_temp = 1.0 / temperature;
    let mut probs: Vec<f32> = indexed
        .iter()
        .map(|&(_, v)| ((v - max_val) * inv_temp).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    let inv_sum = 1.0 / sum;
    for p in probs.iter_mut() {
        *p *= inv_sum;
    }

    let mut rng = rand::thread_rng();
    let r: f32 = rng.gen();
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r <= cumsum {
            return indexed[i].0 as u32;
        }
    }
    indexed.last().unwrap().0 as u32
}

fn top_p_sample(logits: &[f32], p: f32, temperature: f32) -> u32 {
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let n = indexed.len();
    let inv_temp = 1.0 / temperature;
    let max_val = indexed[0].1;
    let mut probs = vec![0.0f32; n];
    let mut sum = 0.0f32;
    for (i, &(_, v)) in indexed.iter().enumerate() {
        let prob = ((v - max_val) * inv_temp).exp();
        probs[i] = prob;
        sum += prob;
    }
    let inv_sum = 1.0 / sum;
    for p in probs.iter_mut() {
        *p *= inv_sum;
    }

    let mut cumsum = 0.0;
    let mut cutoff = n;
    for (i, &prob) in probs.iter().enumerate() {
        cumsum += prob;
        if cumsum >= p {
            cutoff = i + 1;
            break;
        }
    }

    let sum2: f32 = probs[..cutoff].iter().sum();
    let inv_sum2 = 1.0 / sum2;
    for p in probs[..cutoff].iter_mut() {
        *p *= inv_sum2;
    }

    let mut rng = rand::thread_rng();
    let r: f32 = rng.gen();
    let mut cumsum = 0.0;
    for (i, &prob) in probs[..cutoff].iter().enumerate() {
        cumsum += prob;
        if r <= cumsum {
            return indexed[i].0 as u32;
        }
    }
    indexed[cutoff - 1].0 as u32
}

pub fn softmax(logits: &[f32], temperature: f32) -> Vec<f32> {
    let n = logits.len();
    let max_logit = simd::f32_max(logits);
    let inv_temp = 1.0 / temperature;
    let mut probs = vec![0.0f32; n];
    for i in 0..n {
        probs[i] = (logits[i] - max_logit) * inv_temp;
    }
    simd::f32_exp_inplace(&mut probs);
    let sum = simd::f32_sum(&probs);
    simd::f32_scale_inplace(&mut probs, 1.0 / sum);
    probs
}

fn sample_from_probs(probs: &[f32]) -> u32 {
    let mut rng = rand::thread_rng();
    let r: f32 = rng.gen();
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r <= cumsum {
            return i as u32;
        }
    }
    probs.len() as u32 - 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy() {
        let logits = vec![1.0, 3.0, 2.0, 0.5];
        assert_eq!(greedy_sample(&logits), 1);
    }

    #[test]
    fn test_greedy_all_equal() {
        let logits = vec![1.0, 1.0, 1.0];
        assert_eq!(greedy_sample(&logits), 0);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits, 1.0);
        let sum: f32 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "softmax should sum to 1, got {}",
            sum
        );
    }

    #[test]
    fn test_temperature_affects_distribution() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs_hot = softmax(&logits, 2.0);
        let probs_cold = softmax(&logits, 0.5);
        assert!(
            probs_cold[2] > probs_hot[2],
            "colder temperature should concentrate more on argmax"
        );
    }
}
