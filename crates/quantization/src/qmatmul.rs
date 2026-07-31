use crate::scheme::QuantizedTensor;

/// Fused BIT1 (ternary) matmul using XNOR + LUT for the inner loop.
///
/// Weight is shape [n, k], each byte stores 8 ternary values (bit=1 → +1, bit=0 → -1).
///
/// For each group of 8 input elements we build a small LUT (256 f32 entries) that
/// maps every possible 8-bit sign-match mask to the matching-position magnitude sum.
/// Per output neuron we then XNOR the packed input-sign byte with the packed
/// weight-sign byte, look up the LUT entry, and compute the exact dot contribution
/// in O(1) per group (vs. O(8) per group in the naive approach).
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

    // Fast path for tiny matrices (avoids LUT overhead)
    if k <= 8 && n <= 4 {
        let w_scale = weight.scales[0];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for t in 0..k {
                    let idx = j * k + t;
                    let w = if (data[idx / 8] >> (idx % 8)) & 1 == 1 { w_scale } else { -w_scale };
                    sum += input[i * k + t] * w;
                }
                out[i * n + j] = sum;
            }
        }
        return;
    }

    let w_scale = weight.scales[0];
    out.fill(0.0);

    let mut kk = 0;
    while kk < k {
        let kk_end = (kk + KB).min(k);
        let kk_len = kk_end - kk;

        // Process K-tile in groups of 8 elements
        for chunk_start in (0..kk_len).step_by(8) {
            let chunk_end = (chunk_start + 8).min(kk_len);
            let nbits = chunk_end - chunk_start;

            for i in 0..m {
                let in_row = &input[i * k..];
                let out_row = &mut out[i * n..];

                // Pack input signs and compute magnitudes for this group
                let mut in_byte = 0u8;
                let mut mag = [0.0f32; 8];
                let mut mag_total = 0.0f32;
                for off in 0..nbits {
                    let val = in_row[kk + chunk_start + off];
                    if val > 0.0 {
                        in_byte |= 1 << off;
                    }
                    let abs_val = if val >= 0.0 { val } else { -val };
                    mag[off] = abs_val;
                    mag_total += abs_val;
                }

                // Build LUT for this group: for each possible match mask (0..2^nbits-1),
                // lut[mask] = sum of mag[off] where the mask bit is 1
                let nlut = 1usize << nbits;
                let mut lut = [0.0f32; 256];
                for off in 0..nbits {
                    let step = 1 << off;
                    for mask in (step..nlut).rev() {
                        if (mask & step) != 0 {
                            lut[mask] = lut[mask ^ step] + mag[off];
                        }
                    }
                }

                // Process up to 4 output neurons at once
                let mut j = 0;
                while j < n {
                    let rem = n - j;
                    let ncols = rem.min(4);

                    let mut w0 = 0u8;
                    let mut w1 = 0u8;
                    let mut w2 = 0u8;
                    let mut w3 = 0u8;
                    for off in 0..nbits {
                        let bit_pos = kk + chunk_start + off;
                        let idx0 = j * k + bit_pos;
                        if (data[idx0 / 8] >> (idx0 % 8)) & 1 == 1 {
                            w0 |= 1 << off;
                        }
                        if rem > 1 {
                            let idx1 = (j + 1) * k + bit_pos;
                            if (data[idx1 / 8] >> (idx1 % 8)) & 1 == 1 {
                                w1 |= 1 << off;
                            }
                        }
                        if rem > 2 {
                            let idx2 = (j + 2) * k + bit_pos;
                            if (data[idx2 / 8] >> (idx2 % 8)) & 1 == 1 {
                                w2 |= 1 << off;
                            }
                        }
                        if rem > 3 {
                            let idx3 = (j + 3) * k + bit_pos;
                            if (data[idx3 / 8] >> (idx3 % 8)) & 1 == 1 {
                                w3 |= 1 << off;
                            }
                        }
                    }

                    // XNOR to find matching positions, then LUT-lookup the exact contribution
                    let chunk_mask = !0u8 >> (8 - nbits);
                    let match_mask0 = !(in_byte ^ w0) & chunk_mask;
                    out_row[j] += 2.0 * lut[match_mask0 as usize] - mag_total;

                    if rem > 1 {
                        let match_mask1 = !(in_byte ^ w1) & chunk_mask;
                        out_row[j + 1] += 2.0 * lut[match_mask1 as usize] - mag_total;
                    }
                    if rem > 2 {
                        let match_mask2 = !(in_byte ^ w2) & chunk_mask;
                        out_row[j + 2] += 2.0 * lut[match_mask2 as usize] - mag_total;
                    }
                    if rem > 3 {
                        let match_mask3 = !(in_byte ^ w3) & chunk_mask;
                        out_row[j + 3] += 2.0 * lut[match_mask3 as usize] - mag_total;
                    }

                    j += ncols;
                }
            }
        }

        kk = kk_end;
    }

    if w_scale != 1.0 {
        for v in out.iter_mut() {
            *v *= w_scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ternary::ternary_quantize;
    use bitllm_tensor::Tensor;

    /// Reference BIT1 matmul using float operations (unpack each bit)
    fn bit1_matmul_ref(input: &[f32], weight: &QuantizedTensor, out: &mut [f32], m: usize, k: usize, n: usize) {
        let data = &weight.data;
        let scale = weight.scales[0];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for t in 0..k {
                    let idx = j * k + t;
                    let w = if (data[idx / 8] >> (idx % 8)) & 1 == 1 { scale } else { -scale };
                    sum += input[i * k + t] * w;
                }
                out[i * n + j] = sum;
            }
        }
    }

    #[test]
    fn test_fused_bit1_matmul() {
        // Weight shape [n=2, k=3], input shape [m=1, k=3]
        let a = Tensor::from_slice(&[1.0, -1.0, 0.5], &[1, 3]);
        let b = Tensor::from_slice(&[1.0, -1.0, 2.0, -2.0, 0.5, -0.5], &[2, 3]);
        let b_q = ternary_quantize(&b);

        let n = b_q.shape[0];
        let k = b_q.shape[1];
        let m = 1;
        let mut result = vec![0.0f32; m * n];
        fused_bit1_matmul(a.as_f32_slice(), &b_q, &mut result, m, k, n);

        assert!((result[0] - 5.0).abs() < 1e-4, "got {} expected 5.0", result[0]);
        assert!((result[1] - (-5.0)).abs() < 1e-4, "got {} expected -5.0", result[1]);
    }

    #[test]
    fn test_fused_bit1_matches_reference_various_sizes() {
        let sizes = vec![
            (1, 1), (3, 3), (7, 7), (8, 8), (9, 9),
            (10, 10), (15, 15), (16, 16), (17, 17),
            (32, 32), (64, 64), (128, 128),
            (5, 3), (3, 5), (10, 20), (20, 10),
            (64, 128), (128, 64),
            (100, 100), (200, 200),
        ];

        for &(k, n) in &sizes {
            let m = 3;
            let input_data: Vec<f32> = (0..m * k).map(|i| (i as f32 - (m * k / 2) as f32) / (m * k) as f32 * 2.0).collect();
            let weight_data: Vec<f32> = (0..n * k).map(|i| (i as f32 - (n * k / 2) as f32) / (n * k) as f32 * 2.0).collect();

            let w_tensor = Tensor::from_slice(&weight_data, &[n, k]);
            let w_q = ternary_quantize(&w_tensor);

            let mut xnor_result = vec![0.0f32; m * n];
            fused_bit1_matmul(&input_data, &w_q, &mut xnor_result, m, k, n);

            let mut ref_result = vec![0.0f32; m * n];
            bit1_matmul_ref(&input_data, &w_q, &mut ref_result, m, k, n);

            for idx in 0..m * n {
                let diff = (xnor_result[idx] - ref_result[idx]).abs();
                assert!(
                    diff < 1e-3,
                    "Mismatch at size k={}, n={}, idx={}: XNOR got {}, ref got {}",
                    k, n, idx, xnor_result[idx], ref_result[idx]
                );
            }
        }
    }

    #[test]
    fn test_fused_bit1_non_byte_aligned() {
        let k = 10;
        let n = 6;
        let m = 2;
        let input_data: Vec<f32> = (0..m * k).map(|i| (i as f32 - 5.0) * 0.3).collect();
        let weight_data: Vec<f32> = (0..n * k).map(|i| (i as f32 - 15.0) * 0.2).collect();

        let w_tensor = Tensor::from_slice(&weight_data, &[n, k]);
        let w_q = ternary_quantize(&w_tensor);

        let mut xnor_result = vec![0.0f32; m * n];
        fused_bit1_matmul(&input_data, &w_q, &mut xnor_result, m, k, n);

        let mut ref_result = vec![0.0f32; m * n];
        bit1_matmul_ref(&input_data, &w_q, &mut ref_result, m, k, n);

        for idx in 0..m * n {
            let diff = (xnor_result[idx] - ref_result[idx]).abs();
            assert!(
                diff < 1e-3,
                "Non-byte-aligned mismatch at idx={}: XNOR got {}, ref got {}",
                idx, xnor_result[idx], ref_result[idx]
            );
        }
    }

    #[test]
    fn test_fused_bit1_single_activation() {
        let k = 64;
        let n = 8;
        let m = 1;

        let input_data = vec![1.0f32; k];
        let weight_data: Vec<f32> = (0..n * k).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();

        let w_tensor = Tensor::from_slice(&weight_data, &[n, k]);
        let w_q = ternary_quantize(&w_tensor);

        let mut result = vec![0.0f32; m * n];
        fused_bit1_matmul(&input_data, &w_q, &mut result, m, k, n);

        let mut ref_result = vec![0.0f32; m * n];
        bit1_matmul_ref(&input_data, &w_q, &mut ref_result, m, k, n);

        for j in 0..n {
            let diff = (result[j] - ref_result[j]).abs();
            assert!(
                diff < 1e-3,
                "Single activation mismatch at j={}: got {} expected {}",
                j, result[j], ref_result[j]
            );
        }
    }
}
