#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
mod scalar;

// On x86_64: avx2.rs has its own runtime is_x86_feature_detected! dispatch
// that falls back to scalar inline. On non-x86_64, just use scalar.
#[cfg(target_arch = "x86_64")]
pub use avx2::*;

#[cfg(not(target_arch = "x86_64"))]
pub use scalar::*;

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn test_xnor_popcount_1bit_debug() {
        // 32 bytes, deterministic pattern to debug AVX2 path.
        let a: Vec<u8> = (0..32).map(|i| i as u8).collect();
        let b: Vec<u8> = (0..32).map(|i| (i * 3 + 7) as u8).collect();
        let mut got = vec![0u32; 32];
        xnor_popcount_1bit(&a, &b, &mut got, 256);
        for i in 0..32 {
            let expected = (!a[i] ^ b[i]).count_ones();
            if got[i] != expected {
                eprintln!("byte {}: a={:02x} b={:02x} a^b={:02x} ~={:02x} ref_pop={} got_pop={}",
                          i, a[i], b[i], a[i] ^ b[i], !(a[i] ^ b[i]), expected, got[i]);
            }
            assert_eq!(got[i], expected, "mismatch at byte {}", i);
        }
    }

    #[test]
    fn test_xnor_popcount_1bit() {
        // Each byte packs 8 binary values: 1 = +1, 0 = -1.
        // XNOR gives 1 where bits match.
        let a: Vec<u8> = vec![0b1010_1010, 0b1111_0000];
        let b: Vec<u8> = vec![0b1000_1010, 0b1111_0000];
        let mut popcounts = vec![0u32; 2];
        xnor_popcount_1bit(&a, &b, &mut popcounts, 16);
        // byte 0: 10101010 ^ 10001010 = 00100000, ~ = 11011111, popcount = 7
        assert_eq!(popcounts[0], 7);
        // byte 1: identical, popcount = 8
        assert_eq!(popcounts[1], 8);
    }

    #[test]
    fn test_xnor_popcount_1bit_various_sizes() {
        let mut rng = rand::thread_rng();
        for size in [1usize, 7, 8, 31, 32, 33, 64, 127, 128, 129, 256, 1000] {
            let n_bytes = size.div_ceil(8);
            let a: Vec<u8> = (0..n_bytes).map(|_| rng.gen::<u8>()).collect();
            let b: Vec<u8> = (0..n_bytes).map(|_| rng.gen::<u8>()).collect();
            let mut got = vec![0u32; n_bytes];
            xnor_popcount_1bit(&a, &b, &mut got, size);

            for i in 0..n_bytes {
                let valid_bits = if i == n_bytes - 1 && size % 8 != 0 {
                    size % 8
                } else {
                    8
                };
                let mask = if valid_bits == 8 { 0xFF } else { (1u32 << valid_bits) - 1 } as u8;
                let expected = ((!a[i] ^ b[i]) & mask).count_ones();
                assert_eq!(got[i], expected, "mismatch at byte {} for size {}", i, size);
            }
        }
    }

    fn expected_2bit(a: &[u8], b: &[u8], n: usize) -> Vec<u8> {
        let n_out = n.div_ceil(8);
        let mut out = vec![0u8; n_out];
        for i in 0..n {
            let shift = (i % 4) * 2;
            let bits_a = (a[i / 4] >> shift) & 0x03;
            let bits_b = (b[i / 4] >> shift) & 0x03;
            let w_a = match bits_a {
                0x01 => 1.0f32,
                0x02 => -1.0f32,
                _ => 0.0f32,
            };
            let w_b = match bits_b {
                0x01 => 1.0f32,
                0x02 => -1.0f32,
                _ => 0.0f32,
            };
            if w_a * w_b > 0.0 {
                out[i / 8] |= 1 << (i % 8);
            }
        }
        out
    }

    #[test]
    fn test_xnor_popcount_2bit_basic() {
        // 4 elements per byte. 01 = +1, 10 = -1, others = 0.
        // Byte pattern: [e0(2b), e1(2b), e2(2b), e3(2b)] = low bits first.
        // Element 0 = bits 0-1, element 1 = bits 2-3, etc.
        let a: Vec<u8> = vec![0b01_10_01_10]; // elements: 10, 01, 10, 01 (from low to high)
        let b: Vec<u8> = vec![0b01_10_10_01]; // elements: 01, 10, 10, 10
        let n = 4;
        let mut out = vec![0u8; 1];
        xnor_popcount_2bit(&a, &b, &mut out, n);
        let expected = expected_2bit(&a, &b, n);
        assert_eq!(out, expected);
    }

    #[test]
    fn test_xnor_popcount_2bit_various_sizes() {
        let mut rng = rand::thread_rng();
        for size in [1usize, 7, 8, 31, 32, 33, 64, 127, 128, 129, 256, 1000] {
            let n_packed = size.div_ceil(4);
            let a: Vec<u8> = (0..n_packed).map(|_| rng.gen::<u8>()).collect();
            let b: Vec<u8> = (0..n_packed).map(|_| rng.gen::<u8>()).collect();
            let mut got = vec![0u8; size.div_ceil(8)];
            xnor_popcount_2bit(&a, &b, &mut got, size);
            let expected = expected_2bit(&a, &b, size);
            assert_eq!(got, expected, "mismatch for size {}", size);
        }
    }
}
