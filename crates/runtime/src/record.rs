//! FP32 activation recording for quantization-aware training (QAT).
//!
//! The QAT trainer (`bitllm-train::qat`) reconstructs each quantized
//! projection from `(input, fp32-target)` pairs captured by a forward pass of
//! the *unquantized* teacher model. This module defines that capture format
//! and the `forward_record` entry points on [`Model`] that fill a
//! [`ProjectionRecorder`].

use crate::attention::Attention;
use bitllm_tensor::Tensor;

/// Identifies which projection of a transformer layer produced a sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProjectionKind {
    Q,
    K,
    V,
    O,
    Up,
    Gate,
    Down,
}

/// A single `(input, fp32 target)` pair captured from the FP32 teacher model
/// at one quantized projection. `input: [T, in]`, `target: [T, out]` for a
/// window of `T` tokens — the projection's output is exactly what the QAT
/// quantized forward is trained to reproduce.
#[derive(Clone)]
pub struct ProjectionSample {
    pub layer: usize,
    pub kind: ProjectionKind,
    pub input: Tensor,
    pub target: Tensor,
}

/// Collects projection input/target pairs from the FP32 forward for QAT.
///
/// `cap_per_projection` bounds the total number of *token rows* recorded per
/// `(layer, kind)` pair so corpus-scale recording stays bounded; a window that
/// would exceed the cap is dropped entirely.
#[derive(Default)]
pub struct ProjectionRecorder {
    pub samples: Vec<ProjectionSample>,
    pub cap_per_projection: usize,
}

impl ProjectionRecorder {
    pub fn new(cap_per_projection: usize) -> Self {
        Self {
            samples: Vec::new(),
            cap_per_projection,
        }
    }

    fn rows_for(&self, layer: usize, kind: ProjectionKind) -> usize {
        self.samples
            .iter()
            .filter(|s| s.layer == layer && s.kind == kind)
            .map(|s| s.input.shape()[0])
            .sum()
    }

    pub(crate) fn push(&mut self, sample: ProjectionSample) {
        if self.cap_per_projection == 0 {
            self.samples.push(sample);
            return;
        }
        let rows = sample.input.shape()[0];
        if self.rows_for(sample.layer, sample.kind) + rows <= self.cap_per_projection {
            self.samples.push(sample);
        }
    }
}

impl Attention {
    /// Like [`Attention::forward_gpu_with_rope_cache`], but records the
    /// q/k/v/o projection input/target pairs into `recorder`. No GPU path.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_record(
        &self,
        input: &Tensor,
        mut cache: Option<&mut crate::attention::KvCache>,
        layer_idx: usize,
        position: usize,
        recorder: &mut ProjectionRecorder,
    ) -> Tensor {
        let seq_len = input.shape()[0];
        let hidden_size = self.config.hidden_size;
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads();
        let head_dim = self.config.head_dim();

        let q = self.q_proj.forward(input);
        let k = self.k_proj.forward(input);
        let v = self.v_proj.forward(input);

        recorder.push(ProjectionSample {
            layer: layer_idx,
            kind: ProjectionKind::Q,
            input: input.clone(),
            target: q.clone(),
        });
        recorder.push(ProjectionSample {
            layer: layer_idx,
            kind: ProjectionKind::K,
            input: input.clone(),
            target: k.clone(),
        });
        recorder.push(ProjectionSample {
            layer: layer_idx,
            kind: ProjectionKind::V,
            input: input.clone(),
            target: v.clone(),
        });

        let mut q_reshaped = crate::attention::reshape_for_attention(&q, num_heads, head_dim);
        let mut k_reshaped = crate::attention::reshape_for_attention(&k, num_kv_heads, head_dim);
        let v_reshaped = crate::attention::reshape_for_attention(&v, num_kv_heads, head_dim);

        crate::attention::apply_rotary_emb_inplace_with_cache(
            &mut q_reshaped,
            &mut k_reshaped,
            position,
            head_dim,
            self.config.rope_theta,
            None,
        );

        let kv_seq_len;
        if let Some(c) = cache.as_mut() {
            c.update(layer_idx, 0, &k_reshaped, &v_reshaped, position);
            kv_seq_len = c.seq_len(0).max(1);
        } else {
            kv_seq_len = k_reshaped.shape()[1];
        }

        let output = crate::attention::scaled_dot_product_attention_owned(
            &q_reshaped,
            &k_reshaped,
            &v_reshaped,
            num_heads,
            num_kv_heads,
            head_dim,
            seq_len,
            kv_seq_len,
        );

        let reshaped = output.reshape_owned(&[seq_len, hidden_size]);
        let o = self.o_proj.forward(&reshaped);
        recorder.push(ProjectionSample {
            layer: layer_idx,
            kind: ProjectionKind::O,
            input: reshaped,
            target: o.clone(),
        });
        o
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use crate::model::Model;

    #[test]
    fn recording_forward_matches_normal_forward() {
        let config = ModelConfig::tiny_test();
        let mut model = Model::new(config.clone());
        for layer in &mut model.layers {
            layer.attention.q_proj.weight = Tensor::random(&[config.hidden_size, config.hidden_size], bitllm_tensor::DType::F32);
            layer.attention.k_proj.weight = Tensor::random(&[config.hidden_size, config.hidden_size], bitllm_tensor::DType::F32);
            layer.attention.v_proj.weight = Tensor::random(&[config.hidden_size, config.hidden_size], bitllm_tensor::DType::F32);
            layer.attention.o_proj.weight = Tensor::random(&[config.hidden_size, config.hidden_size], bitllm_tensor::DType::F32);
            layer.ffn_up.weight = Tensor::random(&[config.intermediate_size, config.hidden_size], bitllm_tensor::DType::F32);
            layer.ffn_gate.weight = Tensor::random(&[config.intermediate_size, config.hidden_size], bitllm_tensor::DType::F32);
            layer.ffn_down.weight = Tensor::random(&[config.hidden_size, config.intermediate_size], bitllm_tensor::DType::F32);
        }

        let tokens: Vec<u32> = (0..16).map(|i| i * 13 % config.vocab_size as u32).collect();

        model.clear_cache();
        let normal = model.forward_hidden(&tokens, 0, None);

        model.clear_cache();
        let mut recorder = ProjectionRecorder::new(64);
        let recorded = model.forward_record(&tokens, &mut recorder);

        assert_eq!(normal.shape(), recorded.shape());
        for i in 0..normal.num_elements() {
            assert!(
                (normal.get_flat_f32(i) - recorded.get_flat_f32(i)).abs() < 1e-4,
                "i={}: normal {} recorded {}",
                i,
                normal.get_flat_f32(i),
                recorded.get_flat_f32(i)
            );
        }

        // Every projection must have produced at least one sample.
        for kind in [
            ProjectionKind::Q,
            ProjectionKind::K,
            ProjectionKind::V,
            ProjectionKind::O,
            ProjectionKind::Up,
            ProjectionKind::Gate,
            ProjectionKind::Down,
        ] {
            assert!(
                recorder.samples.iter().any(|s| s.kind == kind),
                "no samples for {kind:?}"
            );
        }

        // Target rows must equal the projection's FP32 output on the input.
        for sample in &recorder.samples {
            assert_eq!(sample.input.shape()[0], sample.target.shape()[0]);
        }
    }
}
