use crate::scheme::{OutlierMap, QuantConfig, QuantizedTensor};
use bitllm_tensor::Tensor;
use wide::f32x8;

/// Quantize to packed ternary, keeping the top `outlier_frac` fraction of
/// weights (ranked by |w|) exact. Outlier positions stay in the packed data
/// (sign bit kept) and are corrected at matmul time; see [`OutlierMap`].
///
/// The ternary scale is computed over the **non-outlier** weights: if the
/// outliers defined the scale, a single huge weight would collapse the entire
/// bulk to `±scale` and outlier channels would buy nothing.
pub fn quantize_with_outliers(tensor: &Tensor, outlier_frac: f64) -> QuantizedTensor {
    quantize_grouped_with_outliers(tensor, outlier_frac, 0)
}

/// Group-wise ternary with optional outlier channels.
///
/// With `group_size = 0` this is identical to [`quantize_with_outliers`] (one
/// global scale). With `group_size > 0`, every block of `group_size`
/// consecutive columns along the reduction dim `k` is scaled by its own absmax
/// (outliers excluded), so channel blocks with very different magnitudes no
/// longer collapse to a single `±scale`.
pub fn quantize_grouped_with_outliers(
    tensor: &Tensor,
    outlier_frac: f64,
    group_size: usize,
) -> QuantizedTensor {
    assert!(
        group_size == 0 || group_size.is_multiple_of(8),
        "group_size must be 0 or a multiple of 8 (got {group_size})"
    );

    let src = tensor.to_f32();
    let n = src.num_elements();
    let k = if tensor.shape().len() >= 2 {
        tensor.shape()[1]
    } else {
        1
    };
    let num_groups = if group_size == 0 {
        1
    } else {
        k.div_ceil(group_size)
    };

    let outliers = select_outliers(&src, n, outlier_frac);

    let mut scales = Vec::with_capacity(num_groups);
    for g in 0..num_groups {
        let (lo, hi) = if group_size == 0 {
            (0, k)
        } else {
            let lo = g * group_size;
            let hi = (lo + group_size).min(k);
            (lo, hi)
        };
        let scale = find_absmax_excluding_range(&src, n, k, lo, hi, outliers.as_ref());
        scales.push(scale);
    }

    let mut data = vec![0u8; n.div_ceil(8)];

    for i in 0..n {
        let col = i % k;
        let g = if group_size == 0 { 0 } else { col / group_size };
        let scale = scales[g];
        let inv_scale = if scale == 0.0 { 1.0 } else { 1.0 / scale };
        let v = src.get_flat_f32(i) * inv_scale;
        if v > 0.0 {
            let byte = i / 8;
            let bit = i % 8;
            data[byte] |= 1 << bit;
        }
    }

    let config = if group_size == 0 {
        QuantConfig::ternary_with_outliers(outlier_frac)
    } else {
        QuantConfig::ternary_grouped_with_outliers(outlier_frac, group_size)
    };

    QuantizedTensor {
        data,
        shape: tensor.shape().to_vec(),
        scales,
        config,
        outliers,
    }
}

/// Top-`frac` flat indices by |w| (ties broken by index, deterministic).
fn select_outliers(src: &Tensor, n: usize, frac: f64) -> Option<OutlierMap> {
    if frac <= 0.0 {
        return None;
    }
    let count = ((frac * n as f64).ceil() as usize).min(n);
    if count == 0 {
        return None;
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        let ab = src.get_flat_f32(a).abs();
        let bb = src.get_flat_f32(b).abs();
        bb.partial_cmp(&ab).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut indices = Vec::with_capacity(count);
    let mut values = Vec::with_capacity(count);
    for &i in order.iter().take(count) {
        indices.push(i as u32);
        values.push(src.get_flat_f32(i));
    }

    Some(OutlierMap { indices, values })
}

pub fn ternary_quantize(tensor: &Tensor) -> QuantizedTensor {
    quantize_with_outliers(tensor, 0.0)
}

pub fn ternary_dequantize(qtensor: &QuantizedTensor) -> Tensor {
    let n = qtensor.num_elements();
    let k = if qtensor.shape.len() >= 2 {
        qtensor.shape[1]
    } else {
        1
    };
    let mut result = Tensor::new(&qtensor.shape, bitllm_tensor::DType::F32);

    for i in 0..n {
        let g = if qtensor.config.group_size == 0 {
            0
        } else {
            (i % k) / qtensor.config.group_size
        };
        let scale = qtensor.scales[g];
        let bit = (qtensor.data[i / 8] >> (i % 8)) & 1;
        let val = if bit == 1 { 1.0 } else { -1.0 };
        result.set_flat_f32(i, val * scale);
    }

    if let Some(ref outliers) = qtensor.outliers {
        for (&idx, &value) in outliers.indices.iter().zip(outliers.values.iter()) {
            result.set_flat_f32(idx as usize, value);
        }
    }

    result
}

/// Absmax over flat positions in column range `[lo, hi)` of a row-major
/// `[n/k, k]` tensor, **except** the outlier indices.
fn find_absmax_excluding_range(
    tensor: &Tensor,
    n: usize,
    k: usize,
    lo: usize,
    hi: usize,
    outliers: Option<&OutlierMap>,
) -> f32 {
    let data = tensor.as_f32_slice();
    let num_rows = n / k.max(1);

    match outliers {
        None => {
            // Fast path: no exclusions, use SIMD per row
            let mut max_val = 0.0f32;
            for row in 0..num_rows {
                let start = row * k + lo;
                let end = row * k + hi;
                let row_max = simd_absmax(&data[start..end]);
                if row_max > max_val {
                    max_val = row_max;
                }
            }
            max_val
        }
        Some(o) => {
            // Slow path: need to check exclusions
            let mut excluded = vec![false; n];
            for &idx in &o.indices {
                excluded[idx as usize] = true;
            }

            let mut max_val = 0.0f32;
            for row in 0..num_rows {
                for col in lo..hi {
                    let i = row * k + col;
                    if excluded[i] {
                        continue;
                    }
                    let v = data[i].abs();
                    if v > max_val {
                        max_val = v;
                    }
                }
            }
            max_val
        }
    }
}

/// SIMD-accelerated absmax over a contiguous slice of f32 values.
/// Uses AVX2/NEON when available via the `wide` crate.
#[inline]
fn simd_absmax(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }

    let mut max_vec = f32x8::splat(0.0);
    let chunks = data.chunks_exact(8);

    for chunk in chunks {
        let vec = f32x8::from([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]);
        max_vec = max_vec.max(vec.abs());
    }

    // Extract and find max from the SIMD vector
    let arr: [f32; 8] = max_vec.into();
    let mut max_val = arr[0];
    for &v in &arr[1..] {
        if v > max_val {
            max_val = v;
        }
    }

    // Handle remainder
    for &v in data.chunks_exact(8).remainder() {
        let abs_v = v.abs();
        if abs_v > max_val {
            max_val = abs_v;
        }
    }

    max_val
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ternary_roundtrip() {
        let data = [0.8f32, -0.9, 0.1, -0.6, 0.3, -0.2, 0.7, -0.5];
        let t = Tensor::from_slice(&data, &[8]);
        let qt = ternary_quantize(&t);
        let reconstructed = ternary_dequantize(&qt);

        let scale = qt.scales[0];

        for (i, &orig) in data.iter().enumerate() {
            let recon = reconstructed.get_flat_f32(i);
            if orig > 0.0 {
                assert!(recon > 0.0, "expected positive at {}: got {}", i, recon);
                assert!(
                    (recon - scale).abs() < 0.01,
                    "expected +scale at {}: got {}",
                    i,
                    recon
                );
            } else {
                assert!(recon < 0.0, "expected negative at {}: got {}", i, recon);
                assert!(
                    (recon + scale).abs() < 0.01,
                    "expected -scale at {}: got {}",
                    i,
                    recon
                );
            }
        }
    }

    #[test]
    fn test_ternary_compression() {
        let data: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) / 512.0).collect();
        let t = Tensor::from_slice(&data, &[1024]);
        let qt = ternary_quantize(&t);
        assert!(
            qt.compression_ratio() > 8.0,
            "ternary should compress > 8x, got {}",
            qt.compression_ratio()
        );
    }

    #[test]
    fn test_ternary_all_positive() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let t = Tensor::from_slice(&data, &[4]);
        let qt = ternary_quantize(&t);
        let reconstructed = ternary_dequantize(&qt);
        for i in 0..4 {
            assert!(reconstructed.get_flat_f32(i) > 0.0);
        }
    }

    #[test]
    fn test_ternary_all_negative() {
        let data = [-1.0f32, -2.0, -3.0, -4.0];
        let t = Tensor::from_slice(&data, &[4]);
        let qt = ternary_quantize(&t);
        let reconstructed = ternary_dequantize(&qt);
        for i in 0..4 {
            assert!(reconstructed.get_flat_f32(i) < 0.0);
        }
    }

    #[test]
    fn test_ternary_zero_value() {
        let data = [0.001f32, -0.001];
        let t = Tensor::from_slice(&data, &[2]);
        let qt = ternary_quantize(&t);
        let reconstructed = ternary_dequantize(&qt);
        assert!(reconstructed.get_flat_f32(0) > 0.0);
        assert!(reconstructed.get_flat_f32(1) < 0.0);
    }

    #[test]
    fn test_outlier_roundtrip_reconstructs_exact_values() {
        // 8 weights: the top 25% (2 largest |w|) must round-trip exactly.
        let data = [0.5f32, -0.6, 0.1, -0.2, 0.3, -0.1, 7.0, -9.0];
        let t = Tensor::from_slice(&data, &[8]);
        let qt = quantize_with_outliers(&t, 0.25);
        let reconstructed = ternary_dequantize(&qt);

        // Outlier indices (by |w|): 7 (-9.0), 6 (7.0).
        let outliers = qt.outliers.as_ref().expect("expected outliers");
        assert_eq!(outliers.indices, vec![7, 6]);
        assert_eq!(outliers.values, vec![-9.0, 7.0]);
        assert_eq!(reconstructed.get_flat_f32(7), -9.0);
        assert_eq!(reconstructed.get_flat_f32(6), 7.0);

        // Non-outliers reconstruct to ±scale, where scale is the bulk absmax
        // (outliers excluded): max(0.5, 0.6, 0.1, 0.2, 0.3, 0.1) = 0.6.
        assert_eq!(qt.scales[0], 0.6);
        for (i, &d) in data.iter().enumerate().take(6) {
            let expected = if d > 0.0 { 0.6 } else { -0.6 };
            assert_eq!(reconstructed.get_flat_f32(i), expected);
        }
    }

    #[test]
    fn test_outlier_count_matches_frac() {
        let data: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) * 0.1).collect();
        let t = Tensor::from_slice(&data, &[100]);

        let none = quantize_with_outliers(&t, 0.0);
        assert!(none.outliers.is_none());

        let one_percent = quantize_with_outliers(&t, 0.01);
        assert_eq!(one_percent.outliers.as_ref().unwrap().indices.len(), 1);

        let ten_percent = quantize_with_outliers(&t, 0.1);
        assert_eq!(ten_percent.outliers.as_ref().unwrap().indices.len(), 10);
    }

    #[test]
    fn test_outlier_compression_ratio_stays_high() {
        // Asymptotically, 1% outliers add 8 bytes each: packed data n/8 bytes
        // plus 0.08n bytes of outliers → ~19.5x. A large matrix avoids the
        // per-tensor scale dominating the ratio.
        let data: Vec<f32> = (0..8192).map(|i| (i as f32 - 4096.0) / 4096.0).collect();
        let t = Tensor::from_slice(&data, &[8192]);
        let qt = quantize_with_outliers(&t, 0.01);
        assert!(
            qt.compression_ratio() > 15.0,
            "1% outliers should keep >15x compression, got {}",
            qt.compression_ratio()
        );
    }

    #[test]
    fn test_grouped_quantize_roundtrip() {
        // 2-D tensor [2, 64] with two magnitude bands: first 32 cols ~10x
        // larger than last 32 cols. Grouped quantize with gs=32 should give
        // two scales reflecting each band, and dequantize should reconstruct
        // to ±scale per group.
        let mut data = Vec::with_capacity(128);
        for _row in 0..2 {
            for col in 0..32 {
                data.push((col as f32 - 16.0) * 0.5);
            }
            for col in 0..32 {
                data.push((col as f32 - 16.0) * 0.05);
            }
        }
        let t = Tensor::from_slice(&data, &[2, 64]);
        let qt = quantize_grouped_with_outliers(&t, 0.0, 32);
        assert_eq!(qt.scales.len(), 2);
        assert!(
            qt.scales[0] > 5.0 * qt.scales[1],
            "group 0 should be much larger"
        );

        let reconstructed = ternary_dequantize(&qt);
        for (i, &orig) in data.iter().enumerate() {
            let col = i % 64;
            let g = col / 32;
            let recon = reconstructed.get_flat_f32(i);
            if orig > 0.0 {
                assert!(recon > 0.0, "expected positive at {}: got {}", i, recon);
                assert!(
                    (recon - qt.scales[g]).abs() < 0.01,
                    "expected +scale at {}: got {}",
                    i,
                    recon
                );
            } else {
                assert!(recon < 0.0, "expected negative at {}: got {}", i, recon);
                assert!(
                    (recon + qt.scales[g]).abs() < 0.01,
                    "expected -scale at {}: got {}",
                    i,
                    recon
                );
            }
        }
    }
}
