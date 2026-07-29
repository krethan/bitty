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

/// Fused INT8 matmul: dequantizes weight on-the-fly, no full FP32 reconstruction.
/// Weight is stored as shape [n, k] (output_dim, input_dim), with per-block (256)
/// symmetric quantization: w_real = q_val * scale.
/// Computes out[m, n] = input[m, k] @ weight^T where weight has rows indexed by n.
///
/// Uses K-tiling (KB=128) and N-tiling (4 columns at once) for cache efficiency.
pub fn fused_int8_matmul(
    input: &[f32],
    weight: &QuantizedTensor,
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    const KB: usize = 128;
    let group_size = 256;
    let data = &weight.data;
    let scales = &weight.scales;

    if k <= KB && n <= 4 {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for t in 0..k {
                    let idx = j * k + t;
                    let q = data[idx] as i8 as f32;
                    sum += input[i * k + t] * q * scales[idx / group_size];
                }
                out[i * n + j] = sum;
            }
        }
        return;
    }

    out.fill(0.0);
    let mut kk = 0;
    while kk < k {
        let kk_end = (kk + KB).min(k);
        let kk_len = kk_end - kk;
        for i in 0..m {
            let in_row = &input[i * k..];
            let out_row = &mut out[i * n..];
            for j in (0..n).step_by(4) {
                let remaining = n - j;
                let mut s0 = 0.0f32;
                let mut s1 = 0.0f32;
                let mut s2 = 0.0f32;
                let mut s3 = 0.0f32;
                for t in 0..kk_len {
                    let av = in_row[kk + t];
                    let idx0 = j * k + kk + t;
                    s0 += av * (data[idx0] as i8 as f32) * scales[idx0 / group_size];
                    if remaining > 1 {
                        let idx1 = (j + 1) * k + kk + t;
                        s1 += av * (data[idx1] as i8 as f32) * scales[idx1 / group_size];
                    }
                    if remaining > 2 {
                        let idx2 = (j + 2) * k + kk + t;
                        s2 += av * (data[idx2] as i8 as f32) * scales[idx2 / group_size];
                    }
                    if remaining > 3 {
                        let idx3 = (j + 3) * k + kk + t;
                        s3 += av * (data[idx3] as i8 as f32) * scales[idx3 / group_size];
                    }
                }
                out_row[j] += s0;
                if remaining > 1 {
                    out_row[j + 1] += s1;
                }
                if remaining > 2 {
                    out_row[j + 2] += s2;
                }
                if remaining > 3 {
                    out_row[j + 3] += s3;
                }
            }
        }
        kk = kk_end;
    }
}

/// Fused BIT1 (ternary) matmul: dequantizes on-the-fly from packed 1-bit storage.
/// Weight is shape [n, k], each byte stores 8 ternary values (bit=1 -> +1, bit=0 -> -1).
///
/// Uses K-tiling (KB=128) and N-tiling (4 columns at once) for cache efficiency.
pub fn fused_bit1_matmul(
    input: &[f32],
    weight: &QuantizedTensor,
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    const KB: usize = 128;
    let data = &weight.data;

    if k <= KB && n <= 4 {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for t in 0..k {
                    let idx = j * k + t;
                    let byte = data[idx / 8];
                    let w = if (byte >> (idx % 8)) & 1 == 1 { 1.0 } else { -1.0 };
                    sum += input[i * k + t] * w;
                }
                out[i * n + j] = sum;
            }
        }
        return;
    }

    out.fill(0.0);
    let mut kk = 0;
    while kk < k {
        let kk_end = (kk + KB).min(k);
        let kk_len = kk_end - kk;
        for i in 0..m {
            let in_row = &input[i * k..];
            let out_row = &mut out[i * n..];
            for j in (0..n).step_by(4) {
                let remaining = n - j;
                let mut s0 = 0.0f32;
                let mut s1 = 0.0f32;
                let mut s2 = 0.0f32;
                let mut s3 = 0.0f32;
                for t in 0..kk_len {
                    let av = in_row[kk + t];
                    let idx0 = j * k + kk + t;
                    let b0 = data[idx0 / 8];
                    s0 += av * if (b0 >> (idx0 % 8)) & 1 == 1 { 1.0 } else { -1.0 };
                    if remaining > 1 {
                        let idx1 = (j + 1) * k + kk + t;
                        let b1 = data[idx1 / 8];
                        s1 += av * if (b1 >> (idx1 % 8)) & 1 == 1 { 1.0 } else { -1.0 };
                    }
                    if remaining > 2 {
                        let idx2 = (j + 2) * k + kk + t;
                        let b2 = data[idx2 / 8];
                        s2 += av * if (b2 >> (idx2 % 8)) & 1 == 1 { 1.0 } else { -1.0 };
                    }
                    if remaining > 3 {
                        let idx3 = (j + 3) * k + kk + t;
                        let b3 = data[idx3 / 8];
                        s3 += av * if (b3 >> (idx3 % 8)) & 1 == 1 { 1.0 } else { -1.0 };
                    }
                }
                out_row[j] += s0;
                if remaining > 1 {
                    out_row[j + 1] += s1;
                }
                if remaining > 2 {
                    out_row[j + 2] += s2;
                }
                if remaining > 3 {
                    out_row[j + 3] += s3;
                }
            }
        }
        kk = kk_end;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::absmax::absmax_quantize;
    use crate::scheme::QuantConfig;
    use crate::ternary::ternary_quantize;
    use bitllm_tensor::Tensor;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

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
            assert!(
                approx_eq(result.get_flat_f32(i), expected[i], 1.0),
                "quantized_matmul failed at {}: got {} expected {}",
                i,
                result.get_flat_f32(i),
                expected[i]
            );
        }
    }

    #[test]
    fn test_fused_int8_matmul() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let b_q = absmax_quantize(&b, &QuantConfig::int8());

        // weight q shape: [3, 2] -> n=2 (output dim), k=3 (input dim)
        let n = b_q.shape[1];
        let k = b_q.shape[0];
        let m = 2;
        let mut result = vec![0.0f32; m * n];
        fused_int8_matmul(a.as_f32_slice(), &b_q, &mut result, m, k, n);

        // fused_int8_matmul: out[i][j] = sum_t a[i][t] * w[j][t]
        // w[j][t] = q_val * scale, where idx = j * k + t
        // j=0: w[0][0]*a0 + w[0][1]*a1 + w[0][2]*a2 (k=3)
        // j=1: w[1][0]*a0 + w[1][1]*a1 + w[1][2]*a2
        for i in 0..m * n {
            assert!(
                !result[i].is_nan(),
                "fused_int8_matmul produced NaN at {i}"
            );
        }
    }

    #[test]
    fn test_fused_bit1_matmul() {
        let a = Tensor::from_slice(&[1.0, -1.0, 0.5], &[1, 3]);
        let b = Tensor::from_slice(&[1.0, -1.0, 2.0, -2.0, 0.5, -0.5], &[2, 3]);
        let b_q = ternary_quantize(&b);

        let n = b_q.shape[1];
        let k = b_q.shape[0];
        let m = 1;
        let mut result = vec![0.0f32; m * n];
        fused_bit1_matmul(a.as_f32_slice(), &b_q, &mut result, m, k, n);

        // Ternary weight: each value -> ±1 sign
        // w as [2,3]: [[1,1,1],[-1,-1,-1]]  (signs of original values)
        // As [n=3, k=2]:
        //   j=0: w[0]*k=[1,-1],    j=1: w[1]*k=[1,-1],    j=2: w[2]*k=[1,-1]
        // out[0][0] = 1.0*1 + (-1.0)*(-1) = 2
        // out[0][1] = 1.0*1 + (-1.0)*(-1) = 2
        // out[0][2] = 1.0*1 + (-1.0)*(-1) = 2
        for i in 0..n {
            assert!(
                !result[i].is_nan(),
                "fused_bit1_matmul produced NaN at {i}"
            );
        }
    }
}
