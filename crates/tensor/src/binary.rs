use crate::simd;
use crate::Tensor;

/// Packed binary tensor: each element is +1 or -1, stored as a single bit.
/// 8 elements per byte, LSB first. Per-row scale factor for dequantization.
pub struct BinaryTensor {
    pub data: Vec<u8>,
    pub rows: usize,
    pub cols: usize,
    pub scales: Vec<f32>,
}

impl BinaryTensor {
    /// Quantize an FP32 weight matrix to binary (+1/-1) by sign.
    /// Each output row gets its own scale = max(abs(row)).
    pub fn from_tensor(weight: &Tensor) -> Self {
        let src = weight.to_f32();
        let shape = weight.shape();
        assert_eq!(shape.len(), 2, "BinaryTensor requires 2D tensor");
        let rows = shape[0];
        let cols = shape[1];
        let bytes_per_row = (cols + 7) / 8;

        let mut data = vec![0u8; rows * bytes_per_row];
        let mut scales = Vec::with_capacity(rows);

        for row in 0..rows {
            let mut row_max = 0.0f32;
            for col in 0..cols {
                let v = src.get_flat_f32(row * cols + col).abs();
                if v > row_max {
                    row_max = v;
                }
            }
            let s = if row_max == 0.0 { 1.0 } else { row_max };
            scales.push(s);
            let inv = 1.0 / s;

            for col in 0..cols {
                let v = src.get_flat_f32(row * cols + col) * inv;
                if v > 0.0 {
                    let byte_idx = row * bytes_per_row + col / 8;
                    let bit_idx = col % 8;
                    data[byte_idx] |= 1 << bit_idx;
                }
            }
        }

        BinaryTensor {
            data,
            rows,
            cols,
            scales,
        }
    }

    /// Number of bytes per row.
    pub fn bytes_per_row(&self) -> usize {
        (self.cols + 7) / 8
    }

    /// Total packed data bytes.
    pub fn nbytes(&self) -> usize {
        self.data.len()
    }

    /// Memory savings vs FP32.
    pub fn compression_ratio(&self) -> f64 {
        let fp32_bytes = self.rows * self.cols * 4;
        fp32_bytes as f64 / self.nbytes() as f64
    }

    /// Dequantize back to FP32: unpack sign bits and multiply by per-row scale.
    pub fn dequantize(&self) -> Tensor {
        let bpr = self.bytes_per_row();
        let mut result = Tensor::zeros(&[self.rows, self.cols], crate::DType::F32);
        let out = result.as_f32_slice_mut();

        for row in 0..self.rows {
            let scale = self.scales[row];
            let weight_row = &self.data[row * bpr..row * bpr + bpr];
            for col in 0..self.cols {
                let bit = (weight_row[col / 8] >> (col % 8)) & 1;
                out[row * self.cols + col] = if bit == 1 { scale } else { -scale };
            }
        }

        result
    }

    /// Binary matmul (parallel): output = input @ sign(weight)^T * scale.
    /// Weights are binary (+1/-1), input is FP32.
    /// Uses SIMD f32_dot on unpacked sign vectors, parallelized across output rows.
    ///
    /// `input`: [batch, cols] FP32
    /// returns: [batch, rows] FP32
    pub fn matmul(&self, input: &Tensor) -> Tensor {
        let in_f32 = input.to_f32();
        let batch = input.shape()[0];
        let k = input.shape()[1];
        assert_eq!(k, self.cols);

        let bpr = self.bytes_per_row();
        let mut result = Tensor::zeros(&[batch, self.rows], crate::DType::F32);

        for b_idx in 0..batch {
            let in_row = &in_f32.as_f32_slice()[b_idx * k..(b_idx + 1) * k];
            let out_row = &mut result.as_f32_slice_mut()[b_idx * self.rows..(b_idx + 1) * self.rows];

            // Parallelize across output rows
            use rayon::prelude::*;
            out_row.par_iter_mut().enumerate().for_each(|(row, out_elem)| {
                let weight_row = &self.data[row * bpr..row * bpr + bpr];

                // Unpack weight bits to ±1.0 sign values, 8 at a time, and dot with input
                let mut dot = 0.0f32;
                let mut col = 0;
                while col + 8 <= k {
                    let byte = weight_row[col / 8];
                    let mut signs = [0.0f32; 8];
                    signs[0] = if byte & 0x01 != 0 { 1.0 } else { -1.0 };
                    signs[1] = if byte & 0x02 != 0 { 1.0 } else { -1.0 };
                    signs[2] = if byte & 0x04 != 0 { 1.0 } else { -1.0 };
                    signs[3] = if byte & 0x08 != 0 { 1.0 } else { -1.0 };
                    signs[4] = if byte & 0x10 != 0 { 1.0 } else { -1.0 };
                    signs[5] = if byte & 0x20 != 0 { 1.0 } else { -1.0 };
                    signs[6] = if byte & 0x40 != 0 { 1.0 } else { -1.0 };
                    signs[7] = if byte & 0x80 != 0 { 1.0 } else { -1.0 };
                    dot += simd::f32_dot(&in_row[col..col + 8], &signs);
                    col += 8;
                }
                while col < k {
                    let bit = (weight_row[col / 8] >> (col % 8)) & 1;
                    let w = if bit == 1 { 1.0f32 } else { -1.0f32 };
                    dot += in_row[col] * w;
                    col += 1;
                }

                *out_elem = dot * self.scales[row];
            });
        }

        result
    }

    /// Binary-vs-binary matmul using XNOR+popcount (parallel).
    /// Both input and weights must be binary (+1/-1, sign-based).
    /// This is the fastest path: pure bit operations, no float math in the inner loop.
    ///
    /// `input_binary`: [batch, cols] packed binary (1 bit per element)
    /// `input_n_bits`: actual number of columns
    /// returns: [batch, rows] FP32 (scaled)
    pub fn matmul_binary(&self, input_binary: &[u8], input_n_bits: usize, batch: usize) -> Vec<f32> {
        let k = self.cols;
        assert_eq!(input_n_bits, k);
        let bpr = self.bytes_per_row();
        let in_bytes = (k + 7) / 8;

        use rayon::prelude::*;
        let mut output = vec![0.0f32; batch * self.rows];

        for b_idx in 0..batch {
            let in_bits = &input_binary[b_idx * in_bytes..(b_idx + 1) * in_bytes];
            let out_row = &mut output[b_idx * self.rows..(b_idx + 1) * self.rows];

            out_row.par_iter_mut().enumerate().for_each(|(row, out_elem)| {
                let weight_row = &self.data[row * bpr..row * bpr + bpr];
                let mut popcounts = vec![0u32; in_bytes];
                simd::xnor_popcount_1bit(in_bits, weight_row, &mut popcounts, k);
                let total_pop: u32 = popcounts.iter().sum();
                let sign_dot = 2.0 * total_pop as f32 - k as f32;
                *out_elem = sign_dot * self.scales[row];
            });
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_tensor_creation() {
        let w = Tensor::from_slice(&[1.0, -1.0, 2.0, -2.0, 3.0, -3.0], &[3, 2]);
        let bt = BinaryTensor::from_tensor(&w);
        assert_eq!(bt.rows, 3);
        assert_eq!(bt.cols, 2);
        assert_eq!(bt.scales.len(), 3);
        // Each row has a scale = max(abs(row))
        assert!((bt.scales[0] - 1.0).abs() < 1e-6);
        assert!((bt.scales[1] - 2.0).abs() < 1e-6);
        assert!((bt.scales[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_binary_tensor_matmul_small() {
        let w = Tensor::from_slice(&[1.0, -1.0, -1.0, 1.0], &[2, 2]);
        let bt = BinaryTensor::from_tensor(&w);
        let input = Tensor::from_slice(&[1.0, 1.0], &[1, 2]);
        let out = bt.matmul(&input);
        assert_eq!(out.shape(), &[1, 2]);
    }

    #[test]
    fn test_binary_tensor_compression() {
        let data: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) / 512.0).collect();
        let t = Tensor::from_slice(&data, &[32, 32]);
        let bt = BinaryTensor::from_tensor(&t);
        assert!(
            bt.compression_ratio() > 30.0,
            "binary should compress > 30x, got {}",
            bt.compression_ratio()
        );
    }

    #[test]
    fn test_binary_dequantize() {
        let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 32.0).collect();
        let w = Tensor::from_slice(&data, &[8, 8]);
        let bt = BinaryTensor::from_tensor(&w);
        let recon = bt.dequantize();
        assert_eq!(recon.shape(), &[8, 8]);
        // Every element should be +scale or -scale
        for row in 0..8 {
            let s = bt.scales[row];
            for col in 0..8 {
                let v = recon.get_flat_f32(row * 8 + col);
                assert!(
                    (v - s).abs() < 1e-6 || (v + s).abs() < 1e-6,
                    "dequantize at [{},{}]: got {} expected +/-{}",
                    row, col, v, s
                );
            }
        }
    }

    #[test]
    fn test_binary_matmul_matches_fp32_reference() {
        let data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 128.0).collect();
        let w = Tensor::from_slice(&data, &[8, 32]);
        let bt = BinaryTensor::from_tensor(&w);

        let input_data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) / 16.0).collect();
        let input = Tensor::from_slice(&input_data, &[1, 32]);

        let out_simd = bt.matmul(&input);
        let bpr = bt.bytes_per_row();

        // Manual reference: for each output element, unpack weight signs and dot with input
        for row in 0..bt.rows {
            let mut expected_dot = 0.0f32;
            for col in 0..32 {
                let bit = (bt.data[row * bpr + col / 8] >> (col % 8)) & 1;
                let w_sign = if bit == 1 { 1.0f32 } else { -1.0f32 };
                expected_dot += input_data[col] * w_sign;
            }
            let expected = expected_dot * bt.scales[row];
            let got = out_simd.get_flat_f32(row);
            assert!(
                (got - expected).abs() < 1e-4,
                "binary matmul mismatch at row {}: got {} expected {}",
                row,
                got,
                expected
            );
        }
    }
}
