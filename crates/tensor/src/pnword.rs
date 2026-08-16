use std::arch::x86_64::*;

#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PNActivation256 {
    pub trits: [u64; 4],
    pub active_mask: u128,
    pub zero_run: u32,
    pub flags: u32,
}

#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PNWeight256 {
    pub trits: [u64; 4],
    pub scale: f32,
    pub flags: u32,
}

#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PNActivation512 {
    pub trits: [u64; 8],
    pub active_mask_low: u128,
    pub active_mask_high: u128,
    pub zero_run: u32,
    pub flags: u32,
}

const EVEN_BITS_MASK: u64 = 0x5555_5555_5555_5555;

impl PNActivation256 {
    pub fn pack(values: &[i8]) -> Self {
        let mut trits = [0u64; 4];
        let mut active_mask = 0u128;
        let mut zero_run = 0u32;
        let mut trailing_zeros = true;
        let len = values.len().min(128);
        for (i, val) in values.iter().enumerate().take(len) {
            let val = *val;
            let word_idx = i / 32;
            let trit_idx = i % 32;
            if val != 0 {
                active_mask |= 1u128 << i;
                trailing_zeros = false;
                if val > 0 {
                    trits[word_idx] |= 1u64 << (trit_idx * 2);
                } else {
                    trits[word_idx] |= 1u64 << (trit_idx * 2 + 1);
                }
            } else if trailing_zeros {
                zero_run += 1;
            }
        }
        Self {
            trits,
            active_mask,
            zero_run,
            flags: 0,
        }
    }

    pub fn unpack(&self, out: &mut [i8]) {
        let len = out.len().min(128);
        for (i, o) in out.iter_mut().enumerate().take(len) {
            let word_idx = i / 32;
            let trit_idx = i % 32;
            let pos_bit = (self.trits[word_idx] >> (trit_idx * 2)) & 1;
            let neg_bit = (self.trits[word_idx] >> (trit_idx * 2 + 1)) & 1;
            if pos_bit == 1 {
                *o = 1;
            } else if neg_bit == 1 {
                *o = -1;
            } else {
                *o = 0;
            }
        }
    }

    #[inline]
    pub fn dot(&self, weight: &PNWeight256) -> f32 {
        if self.active_mask == 0 {
            return 0.0;
        }
        let mut pos_matches = 0u32;
        let mut neg_matches = 0u32;
        for i in 0..4 {
            let a = self.trits[i];
            let w = weight.trits[i];
            if a == 0 || w == 0 {
                continue;
            }
            let a_pos = a & EVEN_BITS_MASK;
            let a_neg = (a >> 1) & EVEN_BITS_MASK;
            let w_pos = w & EVEN_BITS_MASK;
            let w_neg = (w >> 1) & EVEN_BITS_MASK;
            let pos = (a_pos & w_pos) | (a_neg & w_neg);
            let neg = (a_pos & w_neg) | (a_neg & w_pos);
            pos_matches += pos.count_ones();
            neg_matches += neg.count_ones();
        }
        (pos_matches as i32 - neg_matches as i32) as f32 * weight.scale
    }

    #[inline]
    pub fn sparsity(&self) -> f32 {
        let active_count = self.active_mask.count_ones();
        1.0 - (active_count as f32 / 128.0)
    }

    #[inline]
    pub fn xor(&self, other: &Self) -> Self {
        if is_x86_feature_detected!("avx2") {
            unsafe { self.xor_avx2(other) }
        } else {
            let mut result_trits = [0u64; 4];
            for (i, r) in result_trits.iter_mut().enumerate() {
                *r = self.trits[i] ^ other.trits[i];
            }
            Self {
                trits: result_trits,
                active_mask: self.active_mask ^ other.active_mask,
                zero_run: 0,
                flags: self.flags,
            }
        }
    }

    #[inline]
    pub fn and(&self, other: &Self) -> Self {
        if is_x86_feature_detected!("avx2") {
            unsafe { self.and_avx2(other) }
        } else {
            let mut result_trits = [0u64; 4];
            for (i, r) in result_trits.iter_mut().enumerate() {
                *r = self.trits[i] & other.trits[i];
            }
            Self {
                trits: result_trits,
                active_mask: self.active_mask & other.active_mask,
                zero_run: 0,
                flags: self.flags,
            }
        }
    }

    #[inline]
    pub fn popcount(&self) -> u32 {
        self.active_mask.count_ones()
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn xor_avx2(&self, other: &Self) -> Self {
        let mut result_trits = [0u64; 4];
        for i in 0..4 {
            let a = _mm_loadu_si128(self.trits.as_ptr().add(i) as *const __m128i);
            let b = _mm_loadu_si128(other.trits.as_ptr().add(i) as *const __m128i);
            _mm_storeu_si128(result_trits.as_mut_ptr().add(i) as *mut __m128i, _mm_xor_si128(a, b));
        }
        Self {
            trits: result_trits,
            active_mask: self.active_mask ^ other.active_mask,
            zero_run: 0,
            flags: self.flags,
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn and_avx2(&self, other: &Self) -> Self {
        let mut result_trits = [0u64; 4];
        for i in 0..4 {
            let a = _mm_loadu_si128(self.trits.as_ptr().add(i) as *const __m128i);
            let b = _mm_loadu_si128(other.trits.as_ptr().add(i) as *const __m128i);
            _mm_storeu_si128(result_trits.as_mut_ptr().add(i) as *mut __m128i, _mm_and_si128(a, b));
        }
        Self {
            trits: result_trits,
            active_mask: self.active_mask & other.active_mask,
            zero_run: 0,
            flags: self.flags,
        }
    }
}

impl PNWeight256 {
    pub fn pack(values: &[i8], scale: f32) -> Self {
        let mut trits = [0u64; 4];
        let len = values.len().min(128);
        for (i, val) in values.iter().enumerate().take(len) {
            let val = *val;
            let word_idx = i / 32;
            let trit_idx = i % 32;
            if val > 0 {
                trits[word_idx] |= 1u64 << (trit_idx * 2);
            } else if val < 0 {
                trits[word_idx] |= 1u64 << (trit_idx * 2 + 1);
            }
        }
        Self { trits, scale, flags: 0 }
    }

    #[inline]
    pub fn xor(&self, other: &Self) -> Self {
        if is_x86_feature_detected!("avx2") {
            unsafe { self.xor_avx2(other) }
        } else {
            let mut result_trits = [0u64; 4];
            for (i, r) in result_trits.iter_mut().enumerate() {
                *r = self.trits[i] ^ other.trits[i];
            }
            Self {
                trits: result_trits,
                scale: self.scale,
                flags: self.flags,
            }
        }
    }

    #[inline]
    pub fn and(&self, other: &Self) -> Self {
        if is_x86_feature_detected!("avx2") {
            unsafe { self.and_avx2(other) }
        } else {
            let mut result_trits = [0u64; 4];
            for (i, r) in result_trits.iter_mut().enumerate() {
                *r = self.trits[i] & other.trits[i];
            }
            Self {
                trits: result_trits,
                scale: self.scale,
                flags: self.flags,
            }
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn xor_avx2(&self, other: &Self) -> Self {
        let mut result_trits = [0u64; 4];
        for i in 0..4 {
            let a = _mm_loadu_si128(self.trits.as_ptr().add(i) as *const __m128i);
            let b = _mm_loadu_si128(other.trits.as_ptr().add(i) as *const __m128i);
            _mm_storeu_si128(result_trits.as_mut_ptr().add(i) as *mut __m128i, _mm_xor_si128(a, b));
        }
        Self {
            trits: result_trits,
            scale: self.scale,
            flags: self.flags,
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn and_avx2(&self, other: &Self) -> Self {
        let mut result_trits = [0u64; 4];
        for i in 0..4 {
            let a = _mm_loadu_si128(self.trits.as_ptr().add(i) as *const __m128i);
            let b = _mm_loadu_si128(other.trits.as_ptr().add(i) as *const __m128i);
            _mm_storeu_si128(result_trits.as_mut_ptr().add(i) as *mut __m128i, _mm_and_si128(a, b));
        }
        Self {
            trits: result_trits,
            scale: self.scale,
            flags: self.flags,
        }
    }
}

impl PNActivation512 {
    pub fn pack(values: &[i8]) -> Self {
        let mut trits = [0u64; 8];
        let mut active_mask_low = 0u128;
        let mut active_mask_high = 0u128;
        let mut zero_run = 0u32;
        let mut trailing_zeros = true;
        let len = values.len().min(256);
        for (i, val) in values.iter().enumerate().take(len) {
            let val = *val;
            let word_idx = i / 32;
            let trit_idx = i % 32;
            if val != 0 {
                if i < 128 {
                    active_mask_low |= 1u128 << i;
                } else {
                    active_mask_high |= 1u128 << (i - 128);
                }
                trailing_zeros = false;
                if val > 0 {
                    trits[word_idx] |= 1u64 << (trit_idx * 2);
                } else {
                    trits[word_idx] |= 1u64 << (trit_idx * 2 + 1);
                }
            } else if trailing_zeros {
                zero_run += 1;
            }
        }
        Self {
            trits,
            active_mask_low,
            active_mask_high,
            zero_run,
            flags: 0,
        }
    }

    #[inline]
    pub fn xor(&self, other: &Self) -> Self {
        if is_x86_feature_detected!("avx512f") {
            unsafe { self.xor_avx512(other) }
        } else {
            let mut result_trits = [0u64; 8];
            for (i, r) in result_trits.iter_mut().enumerate() {
                *r = self.trits[i] ^ other.trits[i];
            }
            Self {
                trits: result_trits,
                active_mask_low: self.active_mask_low ^ other.active_mask_low,
                active_mask_high: self.active_mask_high ^ other.active_mask_high,
                zero_run: 0,
                flags: self.flags,
            }
        }
    }

    #[inline]
    pub fn and(&self, other: &Self) -> Self {
        if is_x86_feature_detected!("avx512f") {
            unsafe { self.and_avx512(other) }
        } else {
            let mut result_trits = [0u64; 8];
            for (i, r) in result_trits.iter_mut().enumerate() {
                *r = self.trits[i] & other.trits[i];
            }
            Self {
                trits: result_trits,
                active_mask_low: self.active_mask_low & other.active_mask_low,
                active_mask_high: self.active_mask_high & other.active_mask_high,
                zero_run: 0,
                flags: self.flags,
            }
        }
    }

    #[inline]
    pub fn popcount(&self) -> u32 {
        self.active_mask_low.count_ones() + self.active_mask_high.count_ones()
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn xor_avx512(&self, other: &Self) -> Self {
        let mut trits = [0u64; 8];
        for i in 0..8 {
            let a = _mm512_loadu_si512(self.trits.as_ptr().add(i) as *const __m512i);
            let b = _mm512_loadu_si512(other.trits.as_ptr().add(i) as *const __m512i);
            _mm512_storeu_si512(trits.as_mut_ptr().add(i) as *mut __m512i, _mm512_xor_si512(a, b));
        }
        Self {
            trits,
            active_mask_low: self.active_mask_low ^ other.active_mask_low,
            active_mask_high: self.active_mask_high ^ other.active_mask_high,
            zero_run: 0,
            flags: self.flags,
        }
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn and_avx512(&self, other: &Self) -> Self {
        let mut trits = [0u64; 8];
        for i in 0..8 {
            let a = _mm512_loadu_si512(self.trits.as_ptr().add(i) as *const __m512i);
            let b = _mm512_loadu_si512(other.trits.as_ptr().add(i) as *const __m512i);
            _mm512_storeu_si512(trits.as_mut_ptr().add(i) as *mut __m512i, _mm512_and_si512(a, b));
        }
        Self {
            trits,
            active_mask_low: self.active_mask_low & other.active_mask_low,
            active_mask_high: self.active_mask_high & other.active_mask_high,
            zero_run: 0,
            flags: self.flags,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pnactivation256_pack_unpack() {
        let mut values = [0i8; 128];
        values[0] = 1;
        values[1] = -1;
        values[31] = 1;
        values[64] = -1;
        let packed = PNActivation256::pack(&values);
        let mut unpacked = [0i8; 128];
        packed.unpack(&mut unpacked);
        assert_eq!(values, unpacked);
    }

    #[test]
    fn test_pnword256_dot_product() {
        // a = [+1, -1, +1, -1], w = [+1, -1, -1, +1]
        // dot = (+1)(+1) + (-1)(-1) + (+1)(-1) + (-1)(+1) = 1+1-1-1 = 0
        let a = PNActivation256::pack(&[1, -1, 1, -1]);
        let w = PNWeight256::pack(&[1, -1, -1, 1], 2.5);
        assert_eq!(a.dot(&w), 0.0);

        // a = [+1, -1, +1, -1, 0, +1], w = [+1, -1, -1, +1, 0, +1]
        // dot = 1+1-1-1+0+1 = 1, scaled = 2.5
        let a2 = PNActivation256::pack(&[1, -1, 1, -1, 0, 1]);
        let w2 = PNWeight256::pack(&[1, -1, -1, 1, 0, 1], 2.5);
        assert_eq!(a2.dot(&w2), 2.5);
        assert_eq!(a.popcount(), 4);
        assert_eq!(a.popcount(), a2.popcount() - 1);
        assert_eq!(a.sparsity(), 1.0 - (4.0 / 128.0));
    }

    #[test]
    fn test_pnword256_bitwise() {
        let a = PNActivation256::pack(&[1, -1, 0, 1, -1]);
        let b = PNActivation256::pack(&[-1, 1, 1, 0, -1]);
        let _and_result = a.and(&b);
        let _xor_result = a.xor(&b);
        assert_eq!(a.popcount(), 4);
    }

    #[test]
    fn test_pnweight256_bitwise() {
        let a = PNWeight256::pack(&[1, -1, 1, -1, 1], 2.5);
        let b = PNWeight256::pack(&[-1, 1, -1, 1, -1], 2.0);
        let _and_result = a.and(&b);
        let _xor_result = a.xor(&b);
    }

    #[test]
    fn test_pnactivation512_pack_unpack() {
        let mut values = [0i8; 256];
        values[0] = 1;
        values[1] = -1;
        values[255] = 1;
        let packed = PNActivation512::pack(&values);
        assert!(packed.active_mask_low != 0);
        assert_eq!(packed.popcount(), 3);
    }

    #[test]
    fn test_pnactivation512_bitwise() {
        let a = PNActivation512::pack(&[1, -1, 0, 1, -1]);
        let b = PNActivation512::pack(&[-1, 1, 1, 0, -1]);
        let _and_result = a.and(&b);
        let _xor_result = a.xor(&b);
        assert_eq!(a.popcount(), 4);
    }
}