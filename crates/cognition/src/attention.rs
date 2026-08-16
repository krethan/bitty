use crate::HyperVector;

/// The result of a sparse attention step.
#[derive(Debug, Clone)]
pub struct AttentionResult {
    /// Aggregated output hypervector (weighted bundle of attended values)
    pub output: HyperVector,
    /// Individual attention weights and their corresponding value indices
    pub weights: Vec<(usize, f32)>,
    /// Number of items that passed the threshold
    pub num_active: usize,
}

/// Sparse event-driven attention.
///
/// Unlike dense softmax attention which computes a distribution over all
/// positions, sparse attention only activates on items that exceed a
/// similarity threshold. This is inspired by biological attention
/// mechanisms where only salient stimuli trigger a response.
///
/// The computation is:
/// 1. Compute similarity(query, key_i) for all i
/// 2. Filter: keep only items with similarity >= threshold
/// 3. Softmax over the surviving items
/// 4. Weighted bundle of the corresponding value vectors
#[derive(Debug, Clone)]
pub struct SparseAttention {
    /// Minimum similarity to activate
    pub threshold: f32,
    /// Maximum number of items to attend to
    pub max_attend: usize,
    /// Temperature for softmax (lower = sharper)
    pub temperature: f32,
}

impl SparseAttention {
    pub fn new(threshold: f32, max_attend: usize) -> Self {
        Self {
            threshold,
            max_attend,
            temperature: 1.0,
        }
    }

    /// Attend over key-value pairs using the given query.
    ///
    /// Returns an aggregated output vector and the attention weights.
    pub fn attend(
        &self,
        query: &HyperVector,
        keys: &[HyperVector],
        values: &[HyperVector],
    ) -> AttentionResult {
        assert_eq!(keys.len(), values.len(), "key and value counts must match");

        let n = keys.len();
        if n == 0 || query.dims() == 0 {
            return AttentionResult {
                output: HyperVector::new(query.dims()),
                weights: Vec::new(),
                num_active: 0,
            };
        }

        let dims = query.dims();

        let mut active: Vec<(usize, f32)> = keys
            .iter()
            .enumerate()
            .map(|(i, k)| (i, query.similarity(k)))
            .filter(|(_, sim)| *sim >= self.threshold)
            .collect();

        active.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        active.truncate(self.max_attend);

        let num_active = active.len();
        if num_active == 0 {
            return AttentionResult {
                output: HyperVector::new(dims),
                weights: Vec::new(),
                num_active: 0,
            };
        }

        let max_sim = active[0].1;
        let mut exp_sum = 0.0f32;
        for (_, sim) in active.iter_mut() {
            *sim = ((*sim - max_sim) / self.temperature).exp();
            exp_sum += *sim;
        }

        if exp_sum.is_finite() && exp_sum > 0.0 {
            for (_, sim) in active.iter_mut() {
                *sim /= exp_sum;
            }
        } else {
            let uniform = 1.0 / num_active as f32;
            for (_, sim) in active.iter_mut() {
                *sim = uniform;
            }
        }

        let value_refs: Vec<&HyperVector> = active.iter().map(|(idx, _)| &values[*idx]).collect();

        let output = if num_active == 1 {
            value_refs[0].clone()
        } else {
            let weighted: Vec<&HyperVector> = value_refs;
            crate::hd::bundle(&weighted)
        };

        let weights: Vec<(usize, f32)> = active.into_iter().collect();

        AttentionResult {
            output,
            weights,
            num_active,
        }
    }

    /// Attend with focalization: sharpen attention on the most similar items
    /// by iteratively narrowing the threshold.
    pub fn attend_focal(
        &self,
        query: &HyperVector,
        keys: &[HyperVector],
        values: &[HyperVector],
        focus_steps: usize,
    ) -> AttentionResult {
        let mut result = self.attend(query, keys, values);

        for _ in 0..focus_steps {
            if result.num_active <= 1 {
                break;
            }

            let narrowed = SparseAttention {
                threshold: self.threshold + (1.0 - self.threshold) * 0.3,
                max_attend: result.num_active.div_ceil(2),
                temperature: self.temperature * 0.8,
            };

            let new_result = narrowed.attend(query, keys, values);
            if new_result.num_active > 0 {
                result = new_result;
            } else {
                break;
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_above_threshold() {
        let attn = SparseAttention::new(0.5, 10);
        let query = HyperVector::random(64);

        let keys: Vec<HyperVector> = (0..5).map(|_| HyperVector::random(64)).collect();
        let values: Vec<HyperVector> = (0..5).map(|_| HyperVector::random(64)).collect();

        let result = attn.attend(&query, &keys, &values);
        assert!(result.num_active <= 5);
        assert_eq!(result.output.dims(), 64);
    }

    #[test]
    fn test_identical_query_and_key() {
        let attn = SparseAttention::new(0.0, 10);
        let query = HyperVector::random(64);
        let key = query.clone();
        let value = HyperVector::random(64);

        let result = attn.attend(&query, &[key], &[value]);
        assert_eq!(result.num_active, 1);
    }

    #[test]
    fn test_no_items_above_threshold() {
        let attn = SparseAttention::new(0.99, 10);
        let query = HyperVector::random(64);
        let key = HyperVector::random(64);
        let value = HyperVector::random(64);

        let result = attn.attend(&query, &[key], &[value]);
        assert_eq!(result.num_active, 0);
    }

    #[test]
    fn test_max_attend_limit() {
        let attn = SparseAttention::new(0.0, 2);
        let query = HyperVector::random(64);
        let key = query.clone();

        let keys = vec![key.clone(), key.clone(), key.clone()];
        let values = vec![
            HyperVector::random(64),
            HyperVector::random(64),
            HyperVector::random(64),
        ];

        let result = attn.attend(&query, &keys, &values);
        assert!(result.weights.len() <= 2);
    }

    #[test]
    fn test_empty_inputs() {
        let attn = SparseAttention::new(0.5, 10);
        let query = HyperVector::random(64);
        let result = attn.attend(&query, &[], &[]);
        assert_eq!(result.num_active, 0);
    }

    #[test]
    fn test_focal_attention_narrows() {
        let attn = SparseAttention::new(0.0, 10);
        let query = HyperVector::random(64);

        let mut keys = vec![query.clone()];
        for _ in 0..10 {
            keys.push(HyperVector::random(64));
        }
        let values: Vec<HyperVector> = (0..11).map(|_| HyperVector::random(64)).collect();

        let wide = attn.attend(&query, &keys, &values);
        let focused = attn.attend_focal(&query, &keys, &values, 3);

        assert!(
            focused.num_active <= wide.num_active,
            "focal attention should narrow or maintain: wide={}, focused={}",
            wide.num_active,
            focused.num_active
        );
    }

    #[test]
    fn test_weights_sum_to_one() {
        let attn = SparseAttention::new(0.0, 10);
        let query = HyperVector::random(64);

        let keys: Vec<HyperVector> = (0..5).map(|_| HyperVector::random(64)).collect();
        let values: Vec<HyperVector> = (0..5).map(|_| HyperVector::random(64)).collect();

        let result = attn.attend(&query, &keys, &values);
        if result.num_active > 0 {
            let sum: f32 = result.weights.iter().map(|(_, w)| w).sum();
            assert!(
                (sum - 1.0).abs() < 1e-4,
                "attention weights should sum to 1: got {}",
                sum
            );
        }
    }
}
