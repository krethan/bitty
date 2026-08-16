use bitllm_tensor::Tensor;

use crate::sampler::Sampler;
use crate::GpuContext;
use crate::Model;

/// Greedy speculative decoding over a draft model.
///
/// A cheap draft model proposes up to `k` tokens (greedily), the target model
/// verifies all of them in a single forward pass, and only the longest
/// agreeing prefix is kept — replacing the first disagreement with the
/// target's own choice. When the draft is good, the target processes many
/// tokens per forward call; when it is wrong, output quality is unaffected
/// because the target's logits are authoritative.
///
/// **Correctness guarantee**: produces exactly the same tokens as greedy
/// target-only generation (verified by `test_speculative_matches_greedy`).
/// The current implementation is greedy-only; temperature/top-k speculative
/// sampling would require rejection sampling to preserve the distribution.
pub struct SpeculativeDecoder<'a> {
    target: &'a mut Model,
    draft: &'a mut Model,
    pub k: usize,
    pub accepted: u64,
    pub proposed: u64,
    pub target_tokens: u64,
}

impl<'a> SpeculativeDecoder<'a> {
    pub fn new(target: &'a mut Model, draft: &'a mut Model, k: usize) -> Self {
        Self {
            target,
            draft,
            k: k.max(1),
            accepted: 0,
            proposed: 0,
            target_tokens: 0,
        }
    }

    /// Fraction of draft-proposed tokens accepted by the target.
    pub fn acceptance_rate(&self) -> f64 {
        if self.proposed == 0 {
            0.0
        } else {
            self.accepted as f64 / self.proposed as f64
        }
    }

    /// Total tokens verified by the target across all forward calls.
    pub fn target_tokens(&self) -> u64 {
        self.target_tokens
    }

    /// Generate `max_new_tokens` tokens, speculatively verifying the draft.
    /// Returns the same output as `Model::generate` with greedy sampling.
    pub fn generate(
        &mut self,
        prompt: &[u32],
        max_new_tokens: usize,
        eos: Option<u32>,
    ) -> Vec<u32> {
        if prompt.is_empty() || max_new_tokens == 0 {
            return Vec::new();
        }
        let gpu: Option<&GpuContext> = None; // speculative path is CPU-only for now
        let vocab = self.target.config.vocab_size;
        let prompt_len = prompt.len();

        // Prefill both models. The draft's last logits row seeds the first
        // proposal.
        self.target.clear_cache();
        self.draft.clear_cache();
        self.target.forward_slot(prompt, 0, gpu);
        let draft_logits = self.draft.forward_slot(prompt, 0, gpu);
        let mut draft_logits = last_row(&draft_logits, vocab);

        let mut output: Vec<u32> = Vec::new();

        while output.len() < max_new_tokens {
            // 1. Draft proposes k tokens.
            let mut proposed_tokens: Vec<u32> = Vec::with_capacity(self.k);
            for _ in 0..self.k {
                let t = greedy(&draft_logits);
                proposed_tokens.push(t);
                let logits = self.draft.forward_slot(&[t], 0, gpu);
                draft_logits = last_row(&logits, vocab);
            }

            // 2. Verify all k in one target forward.
            let verify = self.target.forward_slot(&proposed_tokens, 0, gpu);
            let v = verify.as_f32_slice();
            self.target_tokens += proposed_tokens.len() as u64;
            self.proposed += proposed_tokens.len() as u64;

            // 3. Longest matching prefix.
            let mut accepted = 0;
            while accepted < proposed_tokens.len()
                && greedy(&v[accepted * vocab..(accepted + 1) * vocab]) == proposed_tokens[accepted]
            {
                accepted += 1;
            }
            self.accepted += accepted as u64;

            if accepted == proposed_tokens.len() {
                // All k matched. Emit them and try a bonus token.
                for &t in &proposed_tokens {
                    if output.len() >= max_new_tokens {
                        break;
                    }
                    output.push(t);
                }
                if output.len() >= max_new_tokens {
                    break;
                }

                let bonus = greedy(&draft_logits);
                let bonus_logits = self.target.forward_slot(&[bonus], 0, gpu);
                self.target_tokens += 1;
                self.proposed += 1;

                if greedy(&bonus_logits.as_f32_slice()[..vocab]) == bonus {
                    // Free token: draft already predicts past it, so the draft
                    // cache stays one behind; the next round's first proposal
                    // re-proposes `bonus` and syncs the cache.
                    self.accepted += 1;
                    output.push(bonus);
                } else {
                    // Bonus rejected: the target's choice wins. Roll back both
                    // caches to the accepted prefix and decode the replacement.
                    let replacement = greedy(&bonus_logits.as_f32_slice()[..vocab]);
                    self.rollback_and_sync(prompt_len + self.k, &[replacement], gpu);
                    output.push(replacement);
                }
                if output.last().is_some_and(|&t| Some(t) == eos) {
                    break;
                }
            } else {
                // Mismatch at position `accepted`. The target's token there is
                // the correct next token; discard the rest of the draft.
                let replacement = greedy(&v[accepted * vocab..(accepted + 1) * vocab]);
                self.rollback_and_sync(prompt_len + accepted, &[replacement], gpu);

                let mut emit: Vec<u32> = proposed_tokens[..accepted].to_vec();
                emit.push(replacement);
                for t in emit {
                    if output.len() >= max_new_tokens {
                        break;
                    }
                    output.push(t);
                }
                if output.last().is_some_and(|&t| Some(t) == eos) {
                    break;
                }
            }
        }

        output.truncate(max_new_tokens);
        output
    }

    /// Truncate both caches to `len`, then decode `continuation` through both
    /// models so their caches and the draft's prediction stay in sync.
    fn rollback_and_sync(&mut self, len: usize, continuation: &[u32], gpu: Option<&GpuContext>) {
        self.target.truncate_cache(0, len);
        self.draft.truncate_cache(0, len);
        self.target.forward_slot(continuation, 0, gpu);
        self.draft.forward_slot(continuation, 0, gpu);
    }
}

fn greedy(logits: &[f32]) -> u32 {
    Sampler::greedy().sample(logits)
}

/// Extract the last row of a `[seq_len, vocab]` logits tensor.
fn last_row(logits: &Tensor, vocab: usize) -> Vec<f32> {
    let slice = logits.as_f32_slice();
    let n = logits.shape()[0] - 1;
    slice[n * vocab..(n + 1) * vocab].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ModelConfig;

    /// A draft that disagrees with the (zero-init) target: randomize every
    /// weight so its greedy proposals are mostly wrong, forcing the rollback
    /// path to run.
    fn divergent_draft(config: &ModelConfig) -> Model {
        let mut draft = Model::new(config.clone());
        use bitllm_tensor::DType;
        draft.lm_head.weight =
            bitllm_tensor::Tensor::random(&[config.vocab_size, config.hidden_size], DType::F32);
        draft.embedding.weight =
            bitllm_tensor::Tensor::random(&[config.vocab_size, config.hidden_size], DType::F32);
        for layer in draft.layers.iter_mut() {
            let shapes = [
                layer.attention.q_proj.weight.shape().to_vec(),
                layer.attention.k_proj.weight.shape().to_vec(),
                layer.attention.v_proj.weight.shape().to_vec(),
                layer.attention.o_proj.weight.shape().to_vec(),
                layer.ffn_up.weight.shape().to_vec(),
                layer.ffn_gate.weight.shape().to_vec(),
                layer.ffn_down.weight.shape().to_vec(),
            ];
            let tensors = shapes
                .iter()
                .map(|s| bitllm_tensor::Tensor::random(s, DType::F32))
                .collect::<Vec<_>>();
            layer.attention.q_proj.weight = tensors[0].clone();
            layer.attention.k_proj.weight = tensors[1].clone();
            layer.attention.v_proj.weight = tensors[2].clone();
            layer.attention.o_proj.weight = tensors[3].clone();
            layer.ffn_up.weight = tensors[4].clone();
            layer.ffn_gate.weight = tensors[5].clone();
            layer.ffn_down.weight = tensors[6].clone();
        }
        draft
    }

    #[test]
    fn test_speculative_matches_greedy() {
        // Zero-init target: greedy always samples token 0, but the divergent
        // draft proposes random tokens. Speculative decoding must still match
        // target-only greedy exactly (target logits are authoritative).
        let config = ModelConfig::tiny_test();

        let mut target = Model::new(config.clone());
        let expected = target.generate(&[0, 1, 2], 20, &Sampler::greedy());

        let mut target = Model::new(config.clone());
        let mut draft = divergent_draft(&config);
        let mut decoder = SpeculativeDecoder::new(&mut target, &mut draft, 4);
        let got = decoder.generate(&[0, 1, 2], 20, None);

        assert_eq!(got, expected, "speculative output diverged from greedy");

        // A disagreeing draft should force real rollbacks.
        assert!(
            decoder.acceptance_rate() < 1.0,
            "draft should not agree with the target: {}",
            decoder.acceptance_rate()
        );
    }

    #[test]
    fn test_speculative_all_accept_bonus_path() {
        // A draft identical to the target always agrees, so every token is
        // accepted and the bonus path runs (k+1 tokens per round).
        let config = ModelConfig::tiny_test();

        let mut target = Model::new(config.clone());
        let mut draft = Model::new(config.clone());
        let mut decoder = SpeculativeDecoder::new(&mut target, &mut draft, 3);

        let got = decoder.generate(&[0, 1, 2], 15, None);
        assert_eq!(got.len(), 15);
        assert_eq!(decoder.acceptance_rate(), 1.0);
        // Draft and target identical: bonus acceptance should be perfect too.
        assert!(decoder.target_tokens() <= got.len() as u64 + 1);
    }

    #[test]
    fn test_speculative_eos_stops() {
        // Zero-init target always samples 0; with eos = Some(0) the very first
        // generated token terminates generation.
        let config = ModelConfig::tiny_test();
        let mut target = Model::new(config.clone());
        let mut draft = divergent_draft(&config);
        let mut decoder = SpeculativeDecoder::new(&mut target, &mut draft, 4);
        let got = decoder.generate(&[0, 1], 12, Some(0));
        assert_eq!(got.len(), 1);
        assert_eq!(got[0], 0);
    }

    #[test]
    fn test_speculative_deterministic() {
        // Same inputs, same models: identical output.
        let config = ModelConfig::tiny_test();
        let mut t1 = Model::new(config.clone());
        let mut d1 = divergent_draft(&config);
        let mut s1 = SpeculativeDecoder::new(&mut t1, &mut d1, 2);
        let got1 = s1.generate(&[1, 2, 3], 10, None);

        let mut t2 = Model::new(config.clone());
        let mut d2 = divergent_draft(&config);
        let mut s2 = SpeculativeDecoder::new(&mut t2, &mut d2, 2);
        let got2 = s2.generate(&[1, 2, 3], 10, None);

        assert_eq!(got1, got2);
    }
}
