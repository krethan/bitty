use crate::scheme::{QuantConfig, QuantizedTensor};
use bitllm_tensor::Tensor;

pub fn ternary_quantize(tensor: &Tensor) -> QuantizedTensor {
    let src = tensor.to_f32();
    let n = src.num_elements();

    let scale = find_absmax(&src, n);
    let inv_scale = if scale == 0.0 { 1.0 } else { 1.0 / scale };

    let mut data = vec![0u8; n.div_ceil(8)];

    for i in 0..n {
        let v = src.get_flat_f32(i) * inv_scale;
        let positive = v > 0.0;
        if positive {
            let byte = i / 8;
            let bit = i % 8;
            data[byte] |= 1 << bit;
        }
    }

    QuantizedTensor {
        data,
        shape: tensor.shape().to_vec(),
        scales: vec![scale],
        config: QuantConfig::ternary(),
    }
}

pub fn ternary_dequantize(qtensor: &QuantizedTensor) -> Tensor {
    let n = qtensor.num_elements();
    let scale = qtensor.scales[0];
    let mut result = Tensor::new(&qtensor.shape, bitllm_tensor::DType::F32);

    for i in 0..n {
        let bit = (qtensor.data[i / 8] >> (i % 8)) & 1;
        let val = if bit == 1 { 1.0 } else { -1.0 };
        result.set_flat_f32(i, val * scale);
    }

    result
}

fn find_absmax(tensor: &Tensor, n: usize) -> f32 {
    let mut max_val: f32 = 0.0;
    for i in 0..n {
        let v = tensor.get_flat_f32(i).abs();
        if v > max_val {
            max_val = v;
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

        for i in 0..8 {
            let orig = data[i];
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
}
