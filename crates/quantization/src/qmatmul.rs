use crate::absmax::absmax_dequantize;
use crate::group::GroupQuantizer;
use crate::scheme::QuantizedTensor;
use crate::ternary::ternary_dequantize;
use bitllm_tensor::Tensor;

pub fn quantized_matmul(
    a: &Tensor,
    b_q: &QuantizedTensor,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    if a.ndim() != 2 || b_q.shape.len() != 2 {
        return Err("quantized_matmul requires 2D tensors".into());
    }
    if a.shape()[1] != b_q.shape[0] {
        return Err(format!("shape mismatch: {:?} x {:?}", a.shape(), b_q.shape).into());
    }

    let b = dequantize_for_matmul(b_q)?;

    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b_q.shape[1];

    let a_f32 = a.to_f32();
    let mut result = Tensor::new(&[m, n], bitllm_tensor::DType::F32);

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for t in 0..k {
                sum += a_f32.get_flat_f32(i * k + t) * b.get_flat_f32(t * n + j);
            }
            result.set_flat_f32(i * n + j, sum);
        }
    }

    Ok(result)
}

fn dequantize_for_matmul(qtensor: &QuantizedTensor) -> Result<Tensor, Box<dyn std::error::Error>> {
    match qtensor.config.bits {
        4 => {
            let group_size = qtensor.config.group_size.unwrap_or(128);
            let q = GroupQuantizer::new(group_size);
            Ok(q.dequantize_int4(qtensor))
        }
        8 => Ok(absmax_dequantize(qtensor)),
        1 => Ok(ternary_dequantize(qtensor)),
        _ => Err(format!("unsupported bits: {}", qtensor.config.bits).into()),
    }
}

pub fn fused_dequant_matmul_int8(
    a: &Tensor,
    b_q: &QuantizedTensor,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    if a.ndim() != 2 || b_q.shape.len() != 2 {
        return Err("fused_dequant_matmul_int8 requires 2D tensors".into());
    }
    if a.shape()[1] != b_q.shape[0] {
        return Err(format!("shape mismatch: {:?} x {:?}", a.shape(), b_q.shape).into());
    }

    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b_q.shape[1];
    let a_f32 = a.to_f32();
    let mut result = Tensor::new(&[m, n], bitllm_tensor::DType::F32);

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for t in 0..k {
                let a_val = a_f32.get_flat_f32(i * k + t);
                let group = t / 256;
                let s = b_q.scales[group.min(b_q.scales.len() - 1)];
                let q = b_q.data[t * n + j] as i8 as f32;
                sum += a_val * q * s;
            }
            result.set_flat_f32(i * n + j, sum);
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::absmax::absmax_quantize;
    use crate::scheme::QuantConfig;

    #[test]
    fn test_quantized_matmul_int8() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let b_q = absmax_quantize(&b, &QuantConfig::int8());

        let result = quantized_matmul(&a, &b_q).unwrap();
        assert_eq!(result.shape(), &[2, 2]);

        let expected = [
            1.0 * 1.0 + 2.0 * 3.0 + 3.0 * 5.0,
            1.0 * 2.0 + 2.0 * 4.0 + 3.0 * 6.0,
            4.0 * 1.0 + 5.0 * 3.0 + 6.0 * 5.0,
            4.0 * 2.0 + 5.0 * 4.0 + 6.0 * 6.0,
        ];

        for i in 0..4 {
            let diff = (result.get_flat_f32(i) - expected[i]).abs();
            assert!(
                diff < 1.0,
                "quantized_matmul failed at {}: got {} expected {}",
                i,
                result.get_flat_f32(i),
                expected[i]
            );
        }
    }

    #[test]
    fn test_fused_dequant_matmul_int8() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let b_q = absmax_quantize(&b, &QuantConfig::int8());

        let result = fused_dequant_matmul_int8(&a, &b_q).unwrap();
        assert_eq!(result.shape(), &[2, 2]);

        let expected = [
            1.0 * 1.0 + 2.0 * 3.0 + 3.0 * 5.0,
            1.0 * 2.0 + 2.0 * 4.0 + 3.0 * 6.0,
            4.0 * 1.0 + 5.0 * 3.0 + 6.0 * 5.0,
            4.0 * 2.0 + 5.0 * 4.0 + 6.0 * 6.0,
        ];

        for i in 0..4 {
            let diff = (result.get_flat_f32(i) - expected[i]).abs();
            assert!(
                diff < 1.0,
                "fused_dequant_matmul_int8 failed at {}: got {} expected {}",
                i,
                result.get_flat_f32(i),
                expected[i]
            );
        }
    }
}
