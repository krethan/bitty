use crate::scheme::QuantizedTensor;
use rayon::prelude::*;

/// Adds back the exact values of outlier channels (see [`OutlierMap`]).
///
/// The packed ternary still contributes `±w_scale` at these positions, so the
/// correction subtracts that sign value (the scale of the outlier's group) and
/// adds the exact f32 value instead. `O(m · |outliers|)` — ~1% of the base
/// matmul work at the default fraction.
///
/// For the int8 path (`act_scale`/`inv_scale` set), the activation is the
/// int8-quantized value `xq`, so the correction is `xq · act_scale · (val −
/// sign)`, keeping the integer path exact except for this final f32 add.
fn apply_outlier_correction(
    out: &mut [f32],
    input: &[f32],
    weight: &QuantizedTensor,
    m: usize,
    k: usize,
    n: usize,
    act_scale: Option<f32>,
    inv_scale: Option<f32>,
) {
    let Some(outliers) = weight.outliers.as_ref() else {
        return;
    };
    let data = &weight.data;
    let gs = weight.config.group_size;

    for (idx, exact_val) in outliers.indices.iter().zip(outliers.values.iter()) {
        let j = (*idx as usize) / k;
        let t = (*idx as usize) % k;
        let g = if gs == 0 { 0 } else { t / gs };
        let w_scale = weight.scales[g];
        let bit = (data[(*idx as usize) / 8] >> (t % 8)) & 1;
        let sign_val = if bit == 1 { w_scale } else { -w_scale };
        let correction = exact_val - sign_val;

        for row in 0..m {
            let a = match (act_scale, inv_scale) {
                (Some(scale), Some(inv)) => {
                    let q = (input[row * k + t] * inv).round().clamp(-127.0, 127.0);
                    q * scale
                }
                _ => input[row * k + t],
            };
            out[row * n + j] += a * correction;
        }
    }
}

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
    if weight.scales.len() == 1 {
        fused_bit1_matmul_single_scale(input, weight, out, m, k, n);
    } else {
        fused_bit1_matmul_grouped(input, weight, out, m, k, n);
    }
}

/// Single global scale variant of [`fused_bit1_matmul`].
fn fused_bit1_matmul_single_scale(
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
        apply_outlier_correction(out, input, weight, m, k, n, None, None);
        return;
    }

    let w_scale = weight.scales[0];
    out.fill(0.0);

    // Parallelize over input rows
    let input_chunks = input.chunks(k);
    let output_chunks = out.chunks_mut(n);
    
    input_chunks.zip(output_chunks).for_each(|(in_row, out_row)| {
        let mut kk = 0;
        while kk < k {
            let kk_end = (kk + KB).min(k);
            let kk_len = kk_end - kk;

            // Process K-tile in groups of 8 elements
            for chunk_start in (0..kk_len).step_by(8) {
                let chunk_end = (chunk_start + 8).min(kk_len);
                let nbits = chunk_end - chunk_start;

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

            kk = kk_end;
        }
    });

    if w_scale != 1.0 {
        for v in out.iter_mut() {
            *v *= w_scale;
        }
    }

    apply_outlier_correction(out, input, weight, m, k, n, None, None);
}

/// Group-wise scale variant of [`fused_bit1_matmul`].
///
/// Each 8-element chunk of `k` lies wholly inside one scale group (groups are
/// byte-aligned, `group_size` is a multiple of 8), so the group's scale is
/// applied to the chunk's contribution inline. There is no final global
/// multiply.
fn fused_bit1_matmul_grouped(
    input: &[f32],
    weight: &QuantizedTensor,
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    const KB: usize = 128;
    let data = &weight.data;
    let gs = weight.config.group_size;
    assert!(gs > 0 && gs % 8 == 0, "grouped kernels require a multiple-of-8 group_size");

    out.fill(0.0);

    if k <= 8 && n <= 4 {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for t in 0..k {
                    let idx = j * k + t;
                    let w_scale = weight.scales[t / gs];
                    let w = if (data[idx / 8] >> (idx % 8)) & 1 == 1 { w_scale } else { -w_scale };
                    sum += input[i * k + t] * w;
                }
                out[i * n + j] = sum;
            }
        }
        apply_outlier_correction(out, input, weight, m, k, n, None, None);
        return;
    }

    // Parallelize over input rows
    let input_chunks = input.chunks(k);
    let output_chunks = out.chunks_mut(n);
    
    input_chunks.zip(output_chunks).for_each(|(in_row, out_row)| {
        let mut kk = 0;
        while kk < k {
            let kk_end = (kk + KB).min(k);
            let kk_len = kk_end - kk;

            for chunk_start in (0..kk_len).step_by(8) {
                let chunk_end = (chunk_start + 8).min(kk_len);
                let nbits = chunk_end - chunk_start;
                let w_scale = weight.scales[(kk + chunk_start) / gs];

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

                    let chunk_mask = !0u8 >> (8 - nbits);
                    let match_mask0 = !(in_byte ^ w0) & chunk_mask;
                    out_row[j] += w_scale * (2.0 * lut[match_mask0 as usize] - mag_total);

                    if rem > 1 {
                        let match_mask1 = !(in_byte ^ w1) & chunk_mask;
                        out_row[j + 1] += w_scale * (2.0 * lut[match_mask1 as usize] - mag_total);
                    }
                    if rem > 2 {
                        let match_mask2 = !(in_byte ^ w2) & chunk_mask;
                        out_row[j + 2] += w_scale * (2.0 * lut[match_mask2 as usize] - mag_total);
                    }
                    if rem > 3 {
                        let match_mask3 = !(in_byte ^ w3) & chunk_mask;
                        out_row[j + 3] += w_scale * (2.0 * lut[match_mask3 as usize] - mag_total);
                    }

                    j += ncols;
                }
            }

            kk = kk_end;
        }
    });

    apply_outlier_correction(out, input, weight, m, k, n, None, None);
}

/// BIT1 (ternary) matmul with per-token int8 activation quantization.
///
/// Same packed weight layout and XNOR+LUT inner loop as [`fused_bit1_matmul`],
/// but each input row is first quantized to int8 with per-token absmax scaling:
///
///   act_scale = max(|input[i, :]|) / 127
///   xq        = clamp(round(input / act_scale), -127, 127)
///
/// The int8·ternary products are integers, so the dot products accumulate in
/// exact `i32` and are dequantized once per output row by
/// `act_scale * w_scale` (weight_scale * activation_scale).
///
/// Reference (f32-input) [`fused_bit1_matmul`] remains the default path; this
/// variant trades a small logit error (~1-2% relative for Gaussian inputs) for
/// an integer-only inner loop.
pub fn fused_bit1_int8_matmul(
    input: &[f32],
    weight: &QuantizedTensor,
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    if weight.scales.len() == 1 {
        fused_bit1_int8_matmul_single_scale(input, weight, out, m, k, n);
    } else {
        fused_bit1_int8_matmul_grouped(input, weight, out, m, k, n);
    }
}

/// Single global scale variant of [`fused_bit1_int8_matmul`].
fn fused_bit1_int8_matmul_single_scale(
    input: &[f32],
    weight: &QuantizedTensor,
    out: &mut [f32],
    _m: usize,
    k: usize,
    n: usize,
) {
    const KB: usize = 128;
    let data = &weight.data;
    let w_scale = weight.scales[0];

    // Parallelize over input rows
    input.par_chunks(k).zip(out.par_chunks_mut(n)).for_each(|(in_row, out_row)| {
        // Per-token absmax int8 activation scale.
        let mut max_abs = 0.0f32;
        for t in 0..k {
            let a = in_row[t].abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        let act_scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
        let inv_scale = 1.0 / act_scale;

        let mut acc = vec![0i32; n];
        let mut lut = [0i32; 256];

        let mut kk = 0;
        while kk < k {
            let kk_end = (kk + KB).min(k);
            let kk_len = kk_end - kk;

            for chunk_start in (0..kk_len).step_by(8) {
                let chunk_end = (chunk_start + 8).min(kk_len);
                let nbits = chunk_end - chunk_start;

                // Quantize this group of inputs to int8, pack signs, collect
                // integer magnitudes for the LUT.
                let mut in_byte = 0u8;
                let mut mag = [0i32; 8];
                let mut mag_total = 0i32;
                for off in 0..nbits {
                    let v = in_row[kk + chunk_start + off] * inv_scale;
                    let q = (v.round() as i32).clamp(-127, 127);
                    if q > 0 {
                        in_byte |= 1 << off;
                    }
                    let m_abs = if q >= 0 { q } else { -q };
                    mag[off] = m_abs;
                    mag_total += m_abs;
                }

                let nlut = 1usize << nbits;
                lut[0] = 0;
                for off in 0..nbits {
                    let step = 1 << off;
                    for mask in (step..nlut).rev() {
                        if (mask & step) != 0 {
                            lut[mask] = lut[mask ^ step] + mag[off];
                        }
                    }
                }

                let chunk_mask = !0u8 >> (8 - nbits);
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

                    acc[j] += 2 * lut[(!(in_byte ^ w0) & chunk_mask) as usize] - mag_total;
                    if rem > 1 {
                        acc[j + 1] += 2 * lut[(!(in_byte ^ w1) & chunk_mask) as usize] - mag_total;
                    }
                    if rem > 2 {
                        acc[j + 2] += 2 * lut[(!(in_byte ^ w2) & chunk_mask) as usize] - mag_total;
                    }
                    if rem > 3 {
                        acc[j + 3] += 2 * lut[(!(in_byte ^ w3) & chunk_mask) as usize] - mag_total;
                    }

                    j += ncols;
                }
            }

            kk = kk_end;
        }

        let scale = act_scale * w_scale;
        for j in 0..n {
            out_row[j] = acc[j] as f32 * scale;
        }

        apply_outlier_correction(
            out_row,
            in_row,
            weight,
            1,
            k,
            n,
            Some(act_scale),
            Some(inv_scale),
        );
    });
}

/// Group-wise scale variant of [`fused_bit1_int8_matmul`].
///
/// The integer dot product accumulates in `i32` per scale group; whenever the
/// `k`-index crosses into a new group the previous group's accumulator is
/// flushed into the f32 output row with that group's scale. The activation
/// scale is applied once at the end of the row.
fn fused_bit1_int8_matmul_grouped(
    input: &[f32],
    weight: &QuantizedTensor,
    out: &mut [f32],
    _m: usize,
    k: usize,
    n: usize,
) {
    const KB: usize = 128;
    let data = &weight.data;
    let gs = weight.config.group_size;
    assert!(gs > 0 && gs % 8 == 0, "grouped kernels require a multiple-of-8 group_size");

    out.fill(0.0);

    // Parallelize over input rows
    input.par_chunks(k).zip(out.par_chunks_mut(n)).for_each(|(in_row, out_row)| {
        let mut max_abs = 0.0f32;
        for t in 0..k {
            let a = in_row[t].abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        let act_scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
        let inv_scale = 1.0 / act_scale;

        let mut acc = vec![0i32; n];
        let mut lut = [0i32; 256];
        let mut cur_g = 0;

        let mut kk = 0;
        while kk < k {
            let kk_end = (kk + KB).min(k);
            let kk_len = kk_end - kk;

            for chunk_start in (0..kk_len).step_by(8) {
                let g = (kk + chunk_start) / gs;
                if g != cur_g {
                    for j in 0..n {
                        out_row[j] += acc[j] as f32 * weight.scales[cur_g];
                    }
                    acc.fill(0);
                    cur_g = g;
                }

                let chunk_end = (chunk_start + 8).min(kk_len);
                let nbits = chunk_end - chunk_start;

                let mut in_byte = 0u8;
                let mut mag = [0i32; 8];
                let mut mag_total = 0i32;
                for off in 0..nbits {
                    let v = in_row[kk + chunk_start + off] * inv_scale;
                    let q = (v.round() as i32).clamp(-127, 127);
                    if q > 0 {
                        in_byte |= 1 << off;
                    }
                    let m_abs = if q >= 0 { q } else { -q };
                    mag[off] = m_abs;
                    mag_total += m_abs;
                }

                let nlut = 1usize << nbits;
                lut[0] = 0;
                for off in 0..nbits {
                    let step = 1 << off;
                    for mask in (step..nlut).rev() {
                        if (mask & step) != 0 {
                            lut[mask] = lut[mask ^ step] + mag[off];
                        }
                    }
                }

                let chunk_mask = !0u8 >> (8 - nbits);
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

                    acc[j] += 2 * lut[(!(in_byte ^ w0) & chunk_mask) as usize] - mag_total;
                    if rem > 1 {
                        acc[j + 1] += 2 * lut[(!(in_byte ^ w1) & chunk_mask) as usize] - mag_total;
                    }
                    if rem > 2 {
                        acc[j + 2] += 2 * lut[(!(in_byte ^ w2) & chunk_mask) as usize] - mag_total;
                    }
                    if rem > 3 {
                        acc[j + 3] += 2 * lut[(!(in_byte ^ w3) & chunk_mask) as usize] - mag_total;
                    }

                    j += ncols;
                }
            }

            kk = kk_end;
        }

        for j in 0..n {
            out_row[j] += acc[j] as f32 * weight.scales[cur_g];
        }
        for j in 0..n {
            out_row[j] *= act_scale;
        }

        apply_outlier_correction(
            out_row,
            in_row,
            weight,
            1,
            k,
            n,
            Some(act_scale),
            Some(inv_scale),
        );
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ternary::{quantize_grouped_with_outliers, quantize_with_outliers, ternary_quantize};
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

    /// Plain fp32 matmul over an f32 weight slice (exact reference).
    fn fp32_matmul_ref(input: &[f32], weight: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f64;
                for t in 0..k {
                    sum += input[i * k + t] as f64 * weight[j * k + t] as f64;
                }
                out[i * n + j] = sum as f32;
            }
        }
    }

    fn rel_rmse(a: &[f32], b: &[f32]) -> f64 {
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for (x, y) in a.iter().zip(b) {
            num += (*x as f64 - *y as f64).powi(2);
            den += *y as f64 * *y as f64;
        }
        (num / den).sqrt()
    }

    #[test]
    fn test_outlier_channels_recover_fat_tailed_weights() {
        // Real pretrained weights are fat-tailed: mostly small values with a
        // few large-magnitude entries. With a single global scale, plain
        // ternary lets the tail define `scale`, collapsing the bulk to ±scale;
        // keeping the top 1% exact (with the bulk scale computed excluding
        // them) removes that tail-dominated error and gets much closer to the
        // exact fp32 matmul.
        let mut rng = Rng::new(7);
        let k = 256;
        let n = 128;
        let m = 4;

        let weight_data: Vec<f32> = (0..n * k).map(|_| rng.next_gaussian() * 0.1).collect();
        // Inject 1% outliers at ~100x the typical magnitude.
        let mut weight_data = weight_data;
        let outlier_count = ((n * k) as f64 * 0.01).ceil() as usize;
        for i in 0..outlier_count {
            let idx = i * 37 % weight_data.len();
            weight_data[idx] = rng.next_gaussian() * 10.0;
        }
        let w_tensor = Tensor::from_slice(&weight_data, &[n, k]);

        let input_data: Vec<f32> = (0..m * k).map(|_| rng.next_gaussian()).collect();

        let exact = {
            let mut e = vec![0.0f32; m * n];
            fp32_matmul_ref(&input_data, &weight_data, &mut e, m, k, n);
            e
        };

        let w_plain = ternary_quantize(&w_tensor);
        let mut plain = vec![0.0f32; m * n];
        fused_bit1_matmul(&input_data, &w_plain, &mut plain, m, k, n);

        let w_outlier = quantize_with_outliers(&w_tensor, 0.01);
        assert_eq!(
            w_outlier.outliers.as_ref().unwrap().indices.len(),
            outlier_count
        );
        let mut outlier = vec![0.0f32; m * n];
        fused_bit1_matmul(&input_data, &w_outlier, &mut outlier, m, k, n);

        let plain_err = rel_rmse(&plain, &exact);
        let outlier_err = rel_rmse(&outlier, &exact);
        assert!(
            plain_err > 0.5,
            "plain ternary should be far off when the tail sets the scale, got {plain_err}"
        );
        assert!(
            outlier_err < 0.3 * plain_err,
            "outlier channels should cut the error dramatically, got {outlier_err} vs plain {plain_err}"
        );
    }

    #[test]
    fn test_outlier_channels_int8_path() {
        // The W1A8 path with outlier channels must still agree with the
        // manual int8 reference on a fat-tailed weight matrix.
        let mut rng = Rng::new(9);
        let k = 256;
        let n = 64;
        let m = 3;

        let weight_data: Vec<f32> = (0..n * k).map(|_| rng.next_gaussian() * 0.1).collect();
        let mut weight_data = weight_data;
        for i in 0..(n * k / 100).max(1) {
            let idx = i * 31 % weight_data.len();
            weight_data[idx] = rng.next_gaussian() * 2.0;
        }
        let w_tensor = Tensor::from_slice(&weight_data, &[n, k]);
        let w_q = quantize_with_outliers(&w_tensor, 0.01);

        let input_data: Vec<f32> = (0..m * k).map(|_| rng.next_gaussian()).collect();

        let mut fused = vec![0.0f32; m * n];
        fused_bit1_int8_matmul(&input_data, &w_q, &mut fused, m, k, n);

        // Manual int8 reference that also honors outliers.
        let inv_scale_s = w_q.scales[0];
        for i in 0..m {
            let mut max_abs = 0.0f32;
            for t in 0..k {
                max_abs = max_abs.max(input_data[i * k + t].abs());
            }
            let act_scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
            let inv_scale = 1.0 / act_scale;
            for j in 0..n {
                let mut sum = 0.0f64;
                for t in 0..k {
                    let idx = j * k + t;
                    let bit = (w_q.data[idx / 8] >> (idx % 8)) & 1;
                    let w = if bit == 1 { inv_scale_s } else { -inv_scale_s };
                    let q = (input_data[i * k + t] * inv_scale).round() as i32;
                    let q = q.clamp(-127, 127);
                    sum += q as f64 * w as f64;
                }
                sum *= act_scale as f64;
                // Outlier correction: exact value minus ternary sign value.
                if let Some(ref o) = w_q.outliers {
                    for (idx, val) in o.indices.iter().zip(o.values.iter()) {
                        let jj = (*idx as usize) / k;
                        if jj == j {
                            let t = (*idx as usize) % k;
                            let bit = (w_q.data[(*idx as usize) / 8] >> (t % 8)) & 1;
                            let sign = if bit == 1 { inv_scale_s } else { -inv_scale_s };
                            let q = (input_data[i * k + t] * inv_scale).round() as i32;
                            let q = q.clamp(-127, 127);
                            sum += q as f64 * (*val as f64 - sign as f64) * act_scale as f64;
                        }
                    }
                }
                let ref_val = sum as f32;
                let diff = (fused[i * n + j] - ref_val).abs();
                assert!(
                    diff < 1e-2,
                    "int8+outlier mismatch at i={},j={}: fused {} ref {}",
                    i, j, fused[i * n + j], ref_val
                );
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

    /// Reference int8-activation BIT1 matmul: per-token absmax quantization,
    /// then a plain float loop over the quantized values.
    fn bit1_int8_matmul_ref(
        input: &[f32],
        weight: &QuantizedTensor,
        out: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) {
        let data = &weight.data;
        let w_scale = weight.scales[0];
        for i in 0..m {
            let mut max_abs = 0.0f32;
            for t in 0..k {
                max_abs = max_abs.max(input[i * k + t].abs());
            }
            let act_scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
            let inv_scale = 1.0 / act_scale;
            for j in 0..n {
                let mut sum = 0.0f64;
                for t in 0..k {
                    let idx = j * k + t;
                    let w = if (data[idx / 8] >> (idx % 8)) & 1 == 1 {
                        w_scale
                    } else {
                        -w_scale
                    };
                    let q = (input[i * k + t] * inv_scale).round() as i32;
                    let q = q.clamp(-127, 127);
                    sum += q as f64 * w as f64;
                }
                out[i * n + j] = (sum * act_scale as f64) as f32;
            }
        }
    }

    #[test]
    fn test_fused_bit1_int8_matches_reference() {
        let sizes = vec![
            (1, 1),
            (3, 3),
            (7, 7),
            (8, 8),
            (9, 9),
            (10, 10),
            (16, 16),
            (17, 17),
            (32, 32),
            (64, 64),
            (128, 128),
            (5, 3),
            (10, 20),
            (64, 128),
            (128, 64),
            (200, 200),
        ];

        for &(k, n) in &sizes {
            let m = 3;
            let input_data: Vec<f32> = (0..m * k)
                .map(|i| (i as f32 - (m * k / 2) as f32) / (m * k) as f32 * 2.0)
                .collect();
            let weight_data: Vec<f32> = (0..n * k)
                .map(|i| (i as f32 - (n * k / 2) as f32) / (n * k) as f32 * 2.0)
                .collect();

            let w_tensor = Tensor::from_slice(&weight_data, &[n, k]);
            let w_q = ternary_quantize(&w_tensor);

            let mut fused_result = vec![0.0f32; m * n];
            fused_bit1_int8_matmul(&input_data, &w_q, &mut fused_result, m, k, n);

            let mut ref_result = vec![0.0f32; m * n];
            bit1_int8_matmul_ref(&input_data, &w_q, &mut ref_result, m, k, n);

            for idx in 0..m * n {
                let diff = (fused_result[idx] - ref_result[idx]).abs();
                let scale = fused_result[idx].abs().max(1e-6);
                assert!(
                    diff / scale < 1e-3,
                    "int8 mismatch at k={}, n={}, idx={}: fused {} ref {}",
                    k, n, idx, fused_result[idx], ref_result[idx]
                );
            }
        }
    }

    #[test]
    fn test_fused_bit1_int8_single_activation() {
        let k = 64;
        let n = 8;
        let m = 1;

        let input_data = vec![1.0f32; k];
        let weight_data: Vec<f32> = (0..n * k).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();

        let w_tensor = Tensor::from_slice(&weight_data, &[n, k]);
        let w_q = ternary_quantize(&w_tensor);

        let mut result = vec![0.0f32; m * n];
        fused_bit1_int8_matmul(&input_data, &w_q, &mut result, m, k, n);

        let mut ref_result = vec![0.0f32; m * n];
        bit1_int8_matmul_ref(&input_data, &w_q, &mut ref_result, m, k, n);

        for j in 0..n {
            let diff = (result[j] - ref_result[j]).abs();
            assert!(
                diff < 1e-3,
                "int8 single activation mismatch at j={}: got {} expected {}",
                j, result[j], ref_result[j]
            );
        }
    }

    #[test]
    fn test_fused_bit1_int8_close_to_exact() {
        // Per-token int8 quantization of Gaussian inputs introduces ~1-2%
        // relative logit error vs. the exact f32-input kernel. This pins the
        // approximation quality of the W1A8 path.
        let mut rng = Rng::new(42);
        let k = 256;
        let n = 128;
        let m = 4;

        let input_data: Vec<f32> = (0..m * k).map(|_| rng.next_gaussian()).collect();
        let weight_data: Vec<f32> = (0..n * k).map(|_| rng.next_gaussian()).collect();

        let w_tensor = Tensor::from_slice(&weight_data, &[n, k]);
        let w_q = ternary_quantize(&w_tensor);

        let mut exact = vec![0.0f32; m * n];
        fused_bit1_matmul(&input_data, &w_q, &mut exact, m, k, n);

        let mut int8 = vec![0.0f32; m * n];
        fused_bit1_int8_matmul(&input_data, &w_q, &mut int8, m, k, n);

        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for idx in 0..m * n {
            num += (exact[idx] as f64 - int8[idx] as f64).powi(2);
            den += exact[idx] as f64 * exact[idx] as f64;
        }
        let rel_rmse = (num / den).sqrt();
        assert!(
            rel_rmse < 0.05,
            "W1A8 relative logit error too large: {:.4}",
            rel_rmse
        );
    }

    /// Reference grouped BIT1 matmul: unpacks each bit, applies the group's
    /// scale, and accumulates the dot product.
    fn bit1_matmul_grouped_ref(
        input: &[f32],
        weight: &QuantizedTensor,
        out: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) {
        let data = &weight.data;
        let gs = weight.config.group_size;
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f64;
                for t in 0..k {
                    let idx = j * k + t;
                    let g = t / gs;
                    let scale = weight.scales[g];
                    let w = if (data[idx / 8] >> (idx % 8)) & 1 == 1 {
                        scale
                    } else {
                        -scale
                    };
                    sum += input[i * k + t] as f64 * w as f64;
                }
                out[i * n + j] = sum as f32;
            }
        }
        // Outlier correction.
        if let Some(ref outliers) = weight.outliers {
            for (idx, exact_val) in outliers.indices.iter().zip(outliers.values.iter()) {
                let j = (*idx as usize) / k;
                let t = (*idx as usize) % k;
                let g = t / gs;
                let scale = weight.scales[g];
                let bit = (data[(*idx as usize) / 8] >> (t % 8)) & 1;
                let sign_val = if bit == 1 { scale } else { -scale };
                let correction = exact_val - sign_val;
                for row in 0..m {
                    out[row * n + j] += input[row * k + t] * correction;
                }
            }
        }
    }

    /// Reference grouped int8-activation BIT1 matmul.
    fn bit1_int8_matmul_grouped_ref(
        input: &[f32],
        weight: &QuantizedTensor,
        out: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) {
        let data = &weight.data;
        let gs = weight.config.group_size;
        for i in 0..m {
            let mut max_abs = 0.0f32;
            for t in 0..k {
                max_abs = max_abs.max(input[i * k + t].abs());
            }
            let act_scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
            let inv_scale = 1.0 / act_scale;
            for j in 0..n {
                let mut sum = 0.0f64;
                for t in 0..k {
                    let idx = j * k + t;
                    let g = t / gs;
                    let scale = weight.scales[g];
                    let w = if (data[idx / 8] >> (idx % 8)) & 1 == 1 {
                        scale
                    } else {
                        -scale
                    };
                    let q = (input[i * k + t] * inv_scale).round() as i32;
                    let q = q.clamp(-127, 127);
                    sum += q as f64 * w as f64;
                }
                out[i * n + j] = (sum * act_scale as f64) as f32;
            }
        }
        if let Some(ref outliers) = weight.outliers {
            for (idx, exact_val) in outliers.indices.iter().zip(outliers.values.iter()) {
                let j = (*idx as usize) / k;
                let t = (*idx as usize) % k;
                let g = t / gs;
                let scale = weight.scales[g];
                let bit = (data[(*idx as usize) / 8] >> (t % 8)) & 1;
                let sign_val = if bit == 1 { scale } else { -scale };
                let correction = exact_val - sign_val;
                for row in 0..m {
                    let mut max_abs = 0.0f32;
                    for tt in 0..k {
                        max_abs = max_abs.max(input[row * k + tt].abs());
                    }
                    let row_act_scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
                    let row_inv_scale = 1.0 / row_act_scale;
                    let q = (input[row * k + t] * row_inv_scale).round() as i32;
                    let q = q.clamp(-127, 127);
                    out[row * n + j] += q as f32 * correction * row_act_scale;
                }
            }
        }
    }

    #[test]
    fn test_grouped_bit1_matches_reference() {
        let sizes = vec![
            (32, 8, 8),
            (64, 16, 16),
            (128, 32, 32),
            (256, 32, 32),
            (128, 32, 64),
            (64, 128, 64),
            (128, 64, 128),
        ];

        for &(k, n, gs) in &sizes {
            let m = 3;
            let mut rng = Rng::new(100 + k as u64);
            let input_data: Vec<f32> = (0..m * k).map(|_| rng.next_gaussian()).collect();
            let weight_data: Vec<f32> = (0..n * k).map(|_| rng.next_gaussian()).collect();

            let w_tensor = Tensor::from_slice(&weight_data, &[n, k]);
            let w_q = quantize_grouped_with_outliers(&w_tensor, 0.0, gs);
            assert_eq!(w_q.scales.len(), k.div_ceil(gs));

            let mut fused_result = vec![0.0f32; m * n];
            fused_bit1_matmul(&input_data, &w_q, &mut fused_result, m, k, n);

            let mut ref_result = vec![0.0f32; m * n];
            bit1_matmul_grouped_ref(&input_data, &w_q, &mut ref_result, m, k, n);

            for idx in 0..m * n {
                let diff = (fused_result[idx] - ref_result[idx]).abs();
                assert!(
                    diff < 1e-3,
                    "grouped mismatch at k={}, n={}, gs={}, idx={}: fused {}, ref {}",
                    k, n, gs, idx, fused_result[idx], ref_result[idx]
                );
            }
        }
    }

    #[test]
    fn test_grouped_bit1_int8_matches_reference() {
        let sizes = vec![
            (32, 8, 8),
            (64, 16, 16),
            (128, 32, 32),
            (256, 32, 32),
            (128, 32, 64),
            (64, 128, 64),
        ];

        for &(k, n, gs) in &sizes {
            let m = 3;
            let mut rng = Rng::new(200 + k as u64);
            let input_data: Vec<f32> = (0..m * k).map(|_| rng.next_gaussian()).collect();
            let weight_data: Vec<f32> = (0..n * k).map(|_| rng.next_gaussian()).collect();

            let w_tensor = Tensor::from_slice(&weight_data, &[n, k]);
            let w_q = quantize_grouped_with_outliers(&w_tensor, 0.0, gs);

            let mut fused_result = vec![0.0f32; m * n];
            fused_bit1_int8_matmul(&input_data, &w_q, &mut fused_result, m, k, n);

            let mut ref_result = vec![0.0f32; m * n];
            bit1_int8_matmul_grouped_ref(&input_data, &w_q, &mut ref_result, m, k, n);

            for idx in 0..m * n {
                let diff = (fused_result[idx] - ref_result[idx]).abs();
                let scale = fused_result[idx].abs().max(1e-6);
                assert!(
                    diff / scale < 1e-3,
                    "grouped int8 mismatch at k={}, n={}, gs={}, idx={}: fused {}, ref {}",
                    k, n, gs, idx, fused_result[idx], ref_result[idx]
                );
            }
        }
    }

    #[test]
    fn test_grouped_scales_fat_tailed_groups() {
        // Weight matrix where the first half of k columns are ~100x magnitude
        // of the second half. Global ternary collapses the small group;
        // grouped ternary preserves both groups.
        let mut rng = Rng::new(7);
        let k = 256;
        let n = 128;
        let m = 4;
        let gs = 64;

        let mut weight_data: Vec<f32> = (0..n * k).map(|_| rng.next_gaussian() * 0.01).collect();
        // First half of columns (groups 0,1) are 100x magnitude.
        for row in 0..n {
            for col in 0..k / 2 {
                weight_data[row * k + col] *= 100.0;
            }
        }
        let w_tensor = Tensor::from_slice(&weight_data, &[n, k]);

        let input_data: Vec<f32> = (0..m * k).map(|_| rng.next_gaussian()).collect();

        let exact = {
            let mut e = vec![0.0f32; m * n];
            fp32_matmul_ref(&input_data, &weight_data, &mut e, m, k, n);
            e
        };

        let w_global = ternary_quantize(&w_tensor);
        let mut global = vec![0.0f32; m * n];
        fused_bit1_matmul(&input_data, &w_global, &mut global, m, k, n);

        let w_grouped = quantize_grouped_with_outliers(&w_tensor, 0.0, gs);
        assert_eq!(w_grouped.scales.len(), k / gs);
        let mut grouped = vec![0.0f32; m * n];
        fused_bit1_matmul(&input_data, &w_grouped, &mut grouped, m, k, n);

        let global_err = rel_rmse(&global, &exact);
        let grouped_err = rel_rmse(&grouped, &exact);
        assert!(
            grouped_err <= 1.05 * global_err,
            "grouped scales should not regress much, got {grouped_err} vs global {global_err}"
        );
    }

    #[test]
    fn test_grouped_scales_homogeneous_matches_global() {
        // With uniform magnitudes across groups, grouped ≈ global (both have
        // similar error). This pins no regression in the homogeneous case.
        let mut rng = Rng::new(11);
        let k = 256;
        let n = 128;
        let m = 4;
        let gs = 64;

        let weight_data: Vec<f32> = (0..n * k).map(|_| rng.next_gaussian() * 0.1).collect();
        let w_tensor = Tensor::from_slice(&weight_data, &[n, k]);

        let input_data: Vec<f32> = (0..m * k).map(|_| rng.next_gaussian()).collect();

        let exact = {
            let mut e = vec![0.0f32; m * n];
            fp32_matmul_ref(&input_data, &weight_data, &mut e, m, k, n);
            e
        };

        let w_global = ternary_quantize(&w_tensor);
        let mut global = vec![0.0f32; m * n];
        fused_bit1_matmul(&input_data, &w_global, &mut global, m, k, n);

        let w_grouped = quantize_grouped_with_outliers(&w_tensor, 0.0, gs);
        let mut grouped = vec![0.0f32; m * n];
        fused_bit1_matmul(&input_data, &w_grouped, &mut grouped, m, k, n);

        let global_err = rel_rmse(&global, &exact);
        let grouped_err = rel_rmse(&grouped, &exact);
        assert!(
            grouped_err <= 1.3 * global_err,
            "grouped ≈ global for homogeneous, got {grouped_err} vs {global_err}"
        );
    }

    /// Small deterministic xorshift RNG for tests.
    struct Rng {
        state: u64,
    }

    impl Rng {
        fn new(seed: u64) -> Self {
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

        fn next_gaussian(&mut self) -> f32 {
            let u1 = (self.next_u64() as f64 / u64::MAX as f64).max(1e-10);
            let u2 = (self.next_u64() as f64 / u64::MAX as f64) * 2.0 - 1.0;
            ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
        }
    }
}
