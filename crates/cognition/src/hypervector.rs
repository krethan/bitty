use std::ops::{BitXor, BitAnd, BitOr};

#[derive(Debug, Clone)]
pub struct HyperVector {
    /// Bit storage, LSB-first within each u64
    data: Vec<u64>,
    /// Number of bits (dimensions)
    dims: usize,
}

impl HyperVector {
    pub fn new(dims: usize) -> Self {
        let words = dims.div_ceil(64);
        Self {
            data: vec![0u64; words],
            dims,
        }
    }

    pub fn from_bits(bit_slice: &[bool]) -> Self {
        let dims = bit_slice.len();
        let mut hv = Self::new(dims);
        for (i, &b) in bit_slice.iter().enumerate() {
            if b {
                hv.set_bit(i, true);
            }
        }
        hv
    }

    /// Build from raw LSB-first words (e.g. a packed PNWord trit payload).
    /// The trailing word is masked to `dims` bits.
    pub fn from_words(data: Vec<u64>, dims: usize) -> Self {
        let remainder = dims % 64;
        let mut data = data;
        if remainder > 0 && !data.is_empty() {
            let last = data.len() - 1;
            data[last] &= (1u64 << remainder) - 1;
        }
        Self { data, dims }
    }

    pub fn random(dims: usize) -> Self {
        Self::random_with(dims, &mut rand::thread_rng())
    }

    /// Random balanced vector from an explicit RNG (deterministic for tests
    /// and codebook construction).
    pub fn random_with<R: rand::Rng>(dims: usize, rng: &mut R) -> Self {
        let words = dims.div_ceil(64);
        let mut data = vec![0u64; words];
        for word in data.iter_mut() {
            *word = rng.gen();
        }
        let remainder = dims % 64;
        if remainder > 0 {
            let last = data.len() - 1;
            data[last] &= (1u64 << remainder) - 1;
        }
        Self { data, dims }
    }

    pub fn dims(&self) -> usize {
        self.dims
    }

    pub fn get_bit(&self, i: usize) -> bool {
        assert!(i < self.dims, "bit index out of bounds: {} >= {}", i, self.dims);
        (self.data[i / 64] >> (i % 64)) & 1 == 1
    }

    pub fn set_bit(&mut self, i: usize, val: bool) {
        assert!(i < self.dims, "bit index out of bounds: {} >= {}", i, self.dims);
        let word = i / 64;
        let bit_mask = 1u64 << (i % 64);
        if val {
            self.data[word] |= bit_mask;
        } else {
            self.data[word] &= !bit_mask;
        }
    }

    pub fn flip_bit(&mut self, i: usize) {
        assert!(i < self.dims);
        self.data[i / 64] ^= 1u64 << (i % 64);
    }

    pub fn as_slice(&self) -> &[u64] {
        &self.data
    }

    pub fn hamming_distance(&self, other: &Self) -> u32 {
        assert_eq!(self.dims, other.dims, "dimension mismatch in hamming_distance");
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum()
    }

    pub fn similarity(&self, other: &Self) -> f32 {
        1.0 - (self.hamming_distance(other) as f32 / self.dims as f32)
    }

    pub fn popcount(&self) -> u32 {
        self.data.iter().map(|w| w.count_ones()).sum()
    }

    pub fn density(&self) -> f32 {
        self.popcount() as f32 / self.dims as f32
    }

    pub fn xor(&self, other: &Self) -> Self {
        assert_eq!(self.dims, other.dims, "dimension mismatch in xor");
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a ^ b)
            .collect();
        Self {
            data,
            dims: self.dims,
        }
    }

    pub fn and(&self, other: &Self) -> Self {
        assert_eq!(self.dims, other.dims, "dimension mismatch in and");
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a & b)
            .collect();
        Self {
            data,
            dims: self.dims,
        }
    }

    pub fn or(&self, other: &Self) -> Self {
        assert_eq!(self.dims, other.dims, "dimension mismatch in or");
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a | b)
            .collect();
        Self {
            data,
            dims: self.dims,
        }
    }

    pub fn xnor(&self, other: &Self) -> Self {
        assert_eq!(self.dims, other.dims, "dimension mismatch in xnor");
        let word_len = self.data.len();
        let mut data = vec![0u64; word_len];
        for (i, d) in data.iter_mut().enumerate() {
            *d = !(self.data[i] ^ other.data[i]);
        }
        let remainder = self.dims % 64;
        if remainder > 0 {
            let last = data.len() - 1;
            data[last] &= (1u64 << remainder) - 1;
        }
        Self {
            data,
            dims: self.dims,
        }
    }

    pub fn not(&self) -> Self {
        let word_len = self.data.len();
        let mut data = vec![0u64; word_len];
        for (i, d) in data.iter_mut().enumerate() {
            *d = !self.data[i];
        }
        let remainder = self.dims % 64;
        if remainder > 0 {
            let last = data.len() - 1;
            data[last] &= (1u64 << remainder) - 1;
        }
        Self {
            data,
            dims: self.dims,
        }
    }

    pub fn permute(&self, shift: usize) -> Self {
        let shift = shift % self.dims;
        if shift == 0 {
            return self.clone();
        }
        let n = self.dims;
        let mut result = Self::new(n);
        for i in 0..n {
            let src = (i + n - shift) % n;
            if self.get_bit(src) {
                result.set_bit(i, true);
            }
        }
        result
    }

    pub fn to_f32_slice(&self) -> Vec<f32> {
        (0..self.dims)
            .map(|i| if self.get_bit(i) { 1.0 } else { -1.0 })
            .collect()
    }

    pub fn from_f32_slice(slice: &[f32]) -> Self {
        let mut hv = Self::new(slice.len());
        for (i, &v) in slice.iter().enumerate() {
            if v > 0.0 {
                hv.set_bit(i, true);
            }
        }
        hv
    }
}

impl BitXor for &HyperVector {
    type Output = HyperVector;
    fn bitxor(self, rhs: Self) -> HyperVector {
        self.xor(rhs)
    }
}

impl BitAnd for &HyperVector {
    type Output = HyperVector;
    fn bitand(self, rhs: Self) -> HyperVector {
        self.and(rhs)
    }
}

impl BitOr for &HyperVector {
    type Output = HyperVector;
    fn bitor(self, rhs: Self) -> HyperVector {
        self.or(rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_is_zero() {
        let hv = HyperVector::new(128);
        assert_eq!(hv.dims(), 128);
        assert_eq!(hv.popcount(), 0);
    }

    #[test]
    fn test_random_popcount() {
        let hv = HyperVector::random(512);
        let pc = hv.popcount();
        assert!(pc > 200 && pc < 312, "random popcount out of expected range: {}", pc);
    }

    #[test]
    fn test_set_and_get_bit() {
        let mut hv = HyperVector::new(64);
        assert!(!hv.get_bit(0));
        hv.set_bit(0, true);
        assert!(hv.get_bit(0));
        hv.set_bit(0, false);
        assert!(!hv.get_bit(0));
    }

    #[test]
    fn test_hamming_distance() {
        let a = HyperVector::from_bits(&[true, false, true, false]);
        let b = HyperVector::from_bits(&[true, true, false, false]);
        assert_eq!(a.hamming_distance(&b), 2);
    }

    #[test]
    fn test_similarity() {
        let a = HyperVector::from_bits(&[true, true, true, true]);
        let b = HyperVector::from_bits(&[true, true, true, true]);
        assert!((a.similarity(&b) - 1.0).abs() < 1e-6);
        let c = HyperVector::from_bits(&[true, false, true, false]);
        assert!((a.similarity(&c) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_xor() {
        let a = HyperVector::from_bits(&[true, false, true, false]);
        let b = HyperVector::from_bits(&[true, true, false, false]);
        let x = a.xor(&b);
        assert!(!x.get_bit(0));
        assert!(x.get_bit(1));
        assert!(x.get_bit(2));
        assert!(!x.get_bit(3));
    }

    #[test]
    fn test_xnor() {
        let a = HyperVector::from_bits(&[true, false, true, false]);
        let b = HyperVector::from_bits(&[true, true, false, false]);
        let x = a.xnor(&b);
        assert!(x.get_bit(0));
        assert!(!x.get_bit(1));
        assert!(!x.get_bit(2));
        assert!(x.get_bit(3));
    }

    #[test]
    fn test_permute() {
        let a = HyperVector::from_bits(&[false, true, false, false]);
        let p = a.permute(1);
        assert!(!p.get_bit(0));
        assert!(!p.get_bit(1));
        assert!(p.get_bit(2));
        assert!(!p.get_bit(3));
    }

    #[test]
    fn test_not() {
        let a = HyperVector::from_bits(&[true, false, true, false]);
        let n = a.not();
        assert!(!n.get_bit(0));
        assert!(n.get_bit(1));
        assert!(!n.get_bit(2));
        assert!(n.get_bit(3));
    }

    #[test]
    fn test_from_to_f32_slice() {
        let original = vec![1.0, -1.0, 1.0, -1.0, 1.0];
        let hv = HyperVector::from_f32_slice(&original);
        let recovered = hv.to_f32_slice();
        assert_eq!(recovered, original);
    }

    #[test]
    fn test_xnor_odd_bits() {
        let dims = 7;
        let a = HyperVector::random(dims);
        let b = HyperVector::random(dims);
        let x = a.xnor(&b);
        assert_eq!(x.dims(), dims);
        for i in 0..dims {
            assert_eq!(x.get_bit(i), a.get_bit(i) == b.get_bit(i));
        }
    }

    #[test]
    fn test_bitor() {
        let a = HyperVector::from_bits(&[true, false]);
        let b = HyperVector::from_bits(&[false, true]);
        let o = (&a) | (&b);
        assert!(o.get_bit(0));
        assert!(o.get_bit(1));
    }

    #[test]
    fn test_bitand() {
        let a = HyperVector::from_bits(&[true, false]);
        let b = HyperVector::from_bits(&[true, true]);
        let o = (&a) & (&b);
        assert!(o.get_bit(0));
        assert!(!o.get_bit(1));
    }
}
