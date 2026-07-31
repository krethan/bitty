mod error;
mod init;
pub mod pnword;
pub mod simd;
mod tensor;
mod types;
mod view;

pub use error::TensorError;
pub use init::InitMethod;
pub use tensor::Tensor;
pub use types::DType;
pub use view::TensorView;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Device {
    #[default]
    Cpu,
    Gpu {
        device_id: i32,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_properties() {
        assert_eq!(DType::F32.bit_width(), 32);
        assert_eq!(DType::BIT1.bit_width(), 1);
        assert_eq!(DType::F32.elems_per_byte(), 1);
        assert_eq!(DType::BIT1.elems_per_byte(), 8);
        assert!(DType::F32.is_float());
        assert!(DType::BIT1.is_quantized());
        assert!(!DType::F32.is_quantized());
    }

    #[test]
    fn test_tensor_creation_f32() {
        let t = Tensor::new(&[2, 3], DType::F32);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.num_elements(), 6);
        assert_eq!(t.nbytes(), 24);
        assert_eq!(t.ndim(), 2);
    }

    #[test]
    fn test_tensor_creation_bit1() {
        let t = Tensor::new(&[16], DType::BIT1);
        assert_eq!(t.num_elements(), 16);
        assert_eq!(t.nbytes(), 2);
    }

    #[test]
    fn test_from_slice() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let t = Tensor::from_slice(&data, &[2, 2]);
        assert_eq!(t.get_f32(&[0, 0]), 1.0);
        assert_eq!(t.get_f32(&[0, 1]), 2.0);
        assert_eq!(t.get_f32(&[1, 0]), 3.0);
        assert_eq!(t.get_f32(&[1, 1]), 4.0);
    }

    #[test]
    fn test_zeros_and_ones() {
        let z = Tensor::zeros(&[3], DType::F32);
        for i in 0..3 {
            assert_eq!(z.get_flat_f32(i), 0.0);
        }

        let o = Tensor::ones(&[3], DType::F32);
        for i in 0..3 {
            assert_eq!(o.get_flat_f32(i), 1.0);
        }
    }

    #[test]
    fn test_random_tensor() {
        let t = Tensor::random(&[100], DType::F32);
        assert_eq!(t.num_elements(), 100);
        let mut has_nonzero = false;
        for i in 0..100 {
            let v = t.get_flat_f32(i);
            assert!(v >= -1.0 && v <= 1.0);
            if v != 0.0 {
                has_nonzero = true;
            }
        }
        assert!(has_nonzero);
    }

    #[test]
    fn test_reshape() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let r = t.reshape(&[3, 2]);
        assert_eq!(r.shape(), &[3, 2]);
        assert_eq!(r.get_flat_f32(0), 1.0);
        assert_eq!(r.get_flat_f32(5), 6.0);
    }

    #[test]
    #[should_panic]
    fn test_reshape_wrong_size() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]);
        t.reshape(&[2, 2]);
    }

    #[test]
    fn test_flatten() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let f = t.flatten();
        assert_eq!(f.shape(), &[4]);
        assert_eq!(f.get_flat_f32(0), 1.0);
        assert_eq!(f.get_flat_f32(3), 4.0);
    }

    #[test]
    fn test_transpose() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let tr = t.transpose();
        assert_eq!(tr.shape(), &[3, 2]);
        assert_eq!(tr.get_f32(&[0, 0]), 1.0);
        assert_eq!(tr.get_f32(&[0, 1]), 4.0);
        assert_eq!(tr.get_f32(&[2, 0]), 3.0);
        assert_eq!(tr.get_f32(&[2, 1]), 6.0);
    }

    // === dtype conversion tests ===

    #[test]
    fn test_to_f32_roundtrip() {
        let data = [1.5f32, -2.5, 0.0, 100.0];
        let t = Tensor::from_slice(&data, &[4]);
        let result = t.to_f32();
        for i in 0..4 {
            assert_eq!(result.get_flat_f32(i), data[i]);
        }
    }

    #[test]
    fn test_to_bit1_roundtrip() {
        let data = [1.0f32, -1.0, 1.0, -1.0];
        let t = Tensor::from_slice(&data, &[4]);
        let result = t.to_bit1();
        assert_eq!(result.dtype(), DType::BIT1);
        for i in 0..4 {
            assert_eq!(result.get_flat_f32(i), data[i]);
        }
    }

    #[test]
    fn test_to_bit1_positive_negative() {
        let data = [0.5f32, -0.5, 0.1, -0.9, 0.01, -0.01, 1.0, -1.0];
        let t = Tensor::from_slice(&data, &[8]);
        let result = t.to_bit1();
        for i in 0..8 {
            let expected = if data[i] > 0.0 { 1.0 } else { -1.0 };
            assert_eq!(result.get_flat_f32(i), expected);
        }
    }

    #[test]
    fn test_to_dtype_dispatch() {
        let t = Tensor::from_slice(&[1.0f32], &[1]);
        let b1 = t.to_dtype(DType::BIT1);
        assert_eq!(b1.dtype(), DType::BIT1);
        let f32 = t.to_dtype(DType::F32);
        assert_eq!(f32.dtype(), DType::F32);
    }

    // === arithmetic tests ===

    #[test]
    fn test_add() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = Tensor::from_slice(&[4.0, 5.0, 6.0], &[3]);
        let c = a.add(&b).unwrap();
        assert_eq!(c.get_flat_f32(0), 5.0);
        assert_eq!(c.get_flat_f32(1), 7.0);
        assert_eq!(c.get_flat_f32(2), 9.0);
    }

    #[test]
    fn test_sub() {
        let a = Tensor::from_slice(&[10.0, 20.0], &[2]);
        let b = Tensor::from_slice(&[3.0, 7.0], &[2]);
        let c = a.sub(&b).unwrap();
        assert_eq!(c.get_flat_f32(0), 7.0);
        assert_eq!(c.get_flat_f32(1), 13.0);
    }

    #[test]
    fn test_mul() {
        let a = Tensor::from_slice(&[2.0, 3.0], &[2]);
        let b = Tensor::from_slice(&[4.0, 5.0], &[2]);
        let c = a.mul(&b).unwrap();
        assert_eq!(c.get_flat_f32(0), 8.0);
        assert_eq!(c.get_flat_f32(1), 15.0);
    }

    #[test]
    fn test_add_shape_mismatch() {
        let a = Tensor::from_slice(&[1.0, 2.0], &[2]);
        let b = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]);
        assert!(a.add(&b).is_err());
    }

    #[test]
    fn test_dot() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let c = a.dot(&b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.get_flat_f32(0), 19.0);
        assert_eq!(c.get_flat_f32(1), 22.0);
        assert_eq!(c.get_flat_f32(2), 43.0);
        assert_eq!(c.get_flat_f32(3), 50.0);
    }

    #[test]
    fn test_dot_requires_2d() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]);
        assert!(a.dot(&b).is_err());
    }

    #[test]
    fn test_dot_incompatible_dims() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3, 1]);
        assert!(a.dot(&b).is_err());
    }

    #[test]
    fn test_dot_non_square() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let c = a.dot(&b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.get_flat_f32(0), 22.0);
        assert_eq!(c.get_flat_f32(1), 28.0);
        assert_eq!(c.get_flat_f32(2), 49.0);
        assert_eq!(c.get_flat_f32(3), 64.0);
    }

    // === init tests ===

    #[test]
    fn test_init_constant() {
        let mut t = Tensor::new(&[10], DType::F32);
        InitMethod::Constant(42.0).initialize(&mut t);
        for i in 0..10 {
            assert_eq!(t.get_flat_f32(i), 42.0);
        }
    }

    #[test]
    fn test_init_xavier_normal() {
        let mut t = Tensor::new(&[100], DType::F32);
        InitMethod::XavierNormal.initialize(&mut t);
        let mut sum = 0.0;
        for i in 0..100 {
            sum += t.get_flat_f32(i);
        }
        let mean = sum / 100.0;
        assert!(mean.abs() < 1.0, "XavierNormal mean too large: {}", mean);
    }

    #[test]
    fn test_init_he_normal() {
        let mut t = Tensor::new(&[100], DType::F32);
        InitMethod::HeNormal.initialize(&mut t);
        let mut sum = 0.0;
        for i in 0..100 {
            sum += t.get_flat_f32(i);
        }
        let mean = sum / 100.0;
        assert!(mean.abs() < 1.0, "HeNormal mean too large: {}", mean);
    }

    // === view tests ===

    #[test]
    fn test_tensor_view() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let view = TensorView::new(&t, 0, vec![3, 2], vec![1, 3]);
        assert_eq!(view.shape(), &[3, 2]);
        assert_eq!(view.num_elements(), 6);
        assert_eq!(view.get_f32(&[0, 0]), 1.0);
    }

    #[test]
    fn test_tensor_view_with_offset() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]);
        let view = TensorView::new(&t, 2, vec![3], vec![1]);
        assert_eq!(view.get_f32(&[0]), 3.0);
        assert_eq!(view.get_f32(&[1]), 4.0);
        assert_eq!(view.get_f32(&[2]), 5.0);
    }

    // === edge case tests ===

    #[test]
    fn test_bit1_roundtrip_odd_elements() {
        let data = [1.0f32, -1.0, 1.0, -1.0, 1.0];
        let t = Tensor::from_slice(&data, &[5]);
        let result = t.to_bit1();
        assert_eq!(result.num_elements(), 5);
        for i in 0..5 {
            assert_eq!(result.get_flat_f32(i), data[i]);
        }
    }

    #[test]
    fn test_set_f32_multi_index() {
        let mut t = Tensor::new(&[3, 3], DType::F32);
        t.set_f32(&[1, 2], 42.0);
        assert_eq!(t.get_f32(&[1, 2]), 42.0);
        assert_eq!(t.get_f32(&[0, 0]), 0.0);
    }

    #[test]
    fn test_1d_tensor() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]);
        assert_eq!(t.ndim(), 1);
        assert_eq!(t.get_f32(&[0]), 1.0);
        assert_eq!(t.get_f32(&[2]), 3.0);
    }

    #[test]
    fn test_high_dimensional() {
        let t = Tensor::new(&[2, 2, 2, 2], DType::F32);
        assert_eq!(t.ndim(), 4);
        assert_eq!(t.num_elements(), 16);
        let mut t2 = t;
        t2.set_f32(&[1, 1, 1, 1], 7.0);
        assert_eq!(t2.get_f32(&[1, 1, 1, 1]), 7.0);
    }

    // === SIMD kernel tests ===

    #[test]
    fn test_simd_info() {
        let info = simd::detect_simd_info();
        assert!(!info.is_empty());
        eprintln!("SIMD backend: {}", info);
    }

    #[test]
    fn test_simd_f32_add() {
        let a: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..100).map(|i| i as f32 * 2.0).collect();
        let mut out = vec![0.0f32; 100];
        simd::f32_add(&a, &b, &mut out);
        for i in 0..100 {
            assert!(
                (out[i] - (i as f32 * 3.0)).abs() < 1e-5,
                "f32_add failed at {}: got {} expected {}",
                i,
                out[i],
                i as f32 * 3.0
            );
        }
    }

    #[test]
    fn test_simd_f32_sub() {
        let a: Vec<f32> = (0..100).map(|i| i as f32 * 3.0).collect();
        let b: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let mut out = vec![0.0f32; 100];
        simd::f32_sub(&a, &b, &mut out);
        for i in 0..100 {
            assert!(
                (out[i] - (i as f32 * 2.0)).abs() < 1e-5,
                "f32_sub failed at {}",
                i
            );
        }
    }

    #[test]
    fn test_simd_f32_mul() {
        let a: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..100).map(|i| i as f32 + 1.0).collect();
        let mut out = vec![0.0f32; 100];
        simd::f32_mul(&a, &b, &mut out);
        for i in 0..100 {
            assert!(
                (out[i] - (i as f32 * (i as f32 + 1.0))).abs() < 1e-3,
                "f32_mul failed at {}",
                i
            );
        }
    }

    #[test]
    fn test_simd_f32_dot() {
        let a: Vec<f32> = (0..128).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..128).map(|i| i as f32 * 0.5).collect();
        let result = simd::f32_dot(&a, &b);
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert!(
            (result - expected).abs() < 1e-2,
            "f32_dot failed: got {} expected {}",
            result,
            expected
        );
    }

    #[test]
    fn test_simd_f32_dot_small() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let result = simd::f32_dot(&a, &b);
        assert!((result - 70.0).abs() < 1e-5);
    }

    #[test]
    fn test_simd_f32_sum() {
        let a: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let result = simd::f32_sum(&a);
        let expected: f32 = a.iter().sum();
        assert!((result - expected).abs() < 1e-2);
    }

    #[test]
    fn test_simd_f32_max() {
        let a = vec![1.0, 5.0, 3.0, 9.0, 2.0, -1.0, 7.0, 4.0];
        let result = simd::f32_max(&a);
        assert!((result - 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_simd_f32_scale() {
        let a: Vec<f32> = (0..128).map(|i| i as f32).collect();
        let mut out = vec![0.0f32; 128];
        simd::f32_scale(&a, 2.5, &mut out);
        for i in 0..128 {
            assert!(
                (out[i] - (i as f32 * 2.5)).abs() < 1e-4,
                "f32_scale failed at {}",
                i
            );
        }
    }

    #[test]
    fn test_simd_f32_exp() {
        let a = vec![0.0, 1.0, 2.0, -1.0, 0.5, -0.5, 3.0, -2.0];
        let mut out = vec![0.0f32; 8];
        simd::f32_exp(&a, &mut out);
        for i in 0..8 {
            let expected = a[i].exp();
            assert!(
                (out[i] - expected).abs() < 1e-3,
                "f32_exp failed at {}: got {} expected {}",
                i,
                out[i],
                expected
            );
        }
    }

    #[test]
    fn test_simd_f32_matmul() {
        let m = 4;
        let k = 8;
        let n = 3;
        let a: Vec<f32> = (0..(m * k) as u32).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..(k * n) as u32).map(|i| (i + 1) as f32).collect();
        let mut out = vec![0.0f32; m * n];

        let b_t = transpose_flat(&b, k, n);
        simd::f32_matmul(&a, &b_t, &mut out, m, k, n);

        for i in 0..m {
            for j in 0..n {
                let mut expected = 0.0f32;
                for t in 0..k {
                    expected += a[i * k + t] * b[t * n + j];
                }
                let got = out[i * n + j];
                assert!(
                    (got - expected).abs() < 0.01,
                    "matmul failed at [{},{}]: got {} expected {}",
                    i,
                    j,
                    got,
                    expected
                );
            }
        }
    }

    fn transpose_flat(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut t = vec![0.0f32; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                t[j * rows + i] = a[i * cols + j];
            }
        }
        t
    }

    #[test]
    fn test_simd_i8_dot_product() {
        let a: Vec<u8> = (0..64).map(|i| (i as i8) as u8).collect();
        let b: Vec<u8> = (0..64).map(|i| ((i + 1) as i8) as u8).collect();
        let result = simd::i8_dot_product(&a, &b, 64);
        let expected: i32 = (0..64)
            .map(|i| (a[i] as i8 as i32) * (b[i] as i8 as i32))
            .sum();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_simd_tensor_add_uses_simd() {
        let a = Tensor::from_slice(&(0..256).map(|i| i as f32).collect::<Vec<_>>(), &[256]);
        let b = Tensor::from_slice(
            &(0..256).map(|i| i as f32 * 2.0).collect::<Vec<_>>(),
            &[256],
        );
        let c = a.add(&b).unwrap();
        for i in 0..256 {
            assert!(
                (c.get_flat_f32(i) - (i as f32 * 3.0)).abs() < 1e-4,
                "tensor add SIMD failed at {}",
                i
            );
        }
    }

    #[test]
    fn test_simd_tensor_dot_uses_simd() {
        let a = Tensor::from_slice(&(0..64).map(|i| i as f32).collect::<Vec<_>>(), &[8, 8]);
        let b = Tensor::from_slice(
            &(0..64).map(|i| (i + 1) as f32).collect::<Vec<_>>(),
            &[8, 8],
        );
        let c = a.dot(&b).unwrap();
        assert_eq!(c.shape(), &[8, 8]);

        for i in 0..8 {
            for j in 0..8 {
                let mut expected = 0.0f32;
                for k in 0..8 {
                    expected += a.get_flat_f32(i * 8 + k) * b.get_flat_f32(k * 8 + j);
                }
                let got = c.get_flat_f32(i * 8 + j);
                assert!(
                    (got - expected).abs() < 0.01,
                    "tensor dot SIMD failed at [{},{}]: got {} expected {}",
                    i,
                    j,
                    got,
                    expected
                );
            }
        }
    }
}
