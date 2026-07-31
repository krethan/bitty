use bitllm_tensor::{Tensor, DType};
use bitllm_quantization::ternary_quantize;
use bitllm_quantization::QuantConfig;
use bitllm_quantization::QuantizedTensor;
use bitllm_quantization::fused_bit1_matmul;
use crate::layers::Linear;

/// One-bit (ternary -1/+1) Linear layer.
/// Weights are stored as packed 1-bit sign values, with per-tensor scale.
pub struct BitLinear {
    pub weight_q: QuantizedTensor,
    pub scale: Tensor,
    pub bias: Option<Tensor>,
}

impl BitLinear {
    /// Quantize a full-precision weight tensor to 1-bit ternary.
    pub fn quantize(weight: &Tensor, _config: &QuantConfig) -> Self {
        let weight = ternary_quantize(weight);
        let n = weight.shape[0];
        let scale_val = weight.scales[0];
        let scale_vec = Tensor::from_slice(&vec![scale_val; n], &[n]);
        Self {
            weight_q: weight,
            scale: scale_vec,
            bias: None,
        }
    }

    /// Construct from pre-quantized weights.
    pub fn from_quantized(weight_q: QuantizedTensor, scale: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight_q, scale, bias }
    }

    /// Create a BitLinear from a standard Linear by quantizing its weights.
    pub fn from_linear(linear: &Linear) -> Self {
        let weight_q = ternary_quantize(&linear.weight);
        let n = weight_q.shape[0];
        let scale_val = weight_q.scales[0];
        let scale_vec = Tensor::from_slice(&vec![scale_val; n], &[n]);
        Self {
            weight_q,
            scale: scale_vec,
            bias: linear.bias.clone(),
        }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        let m = input.shape()[0];
        let k = input.shape()[1];
        let n = self.weight_q.shape[0];
        assert_eq!(k, self.weight_q.shape[1]);

        let input_slice = input.as_f32_slice();
        let mut result = Tensor::zeros(&[m, n], DType::F32);
        let out_slice = result.as_f32_slice_mut();

        fused_bit1_matmul(input_slice, &self.weight_q, out_slice, m, k, n);

        let scale = self.weight_q.scales[0];
        if scale != 1.0 {
            for v in out_slice.iter_mut() {
                *v *= scale;
            }
        }

        if let Some(ref bias) = self.bias {
            result.add_assign(bias).unwrap();
        }

        result
    }
}
