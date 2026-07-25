use crate::error::TensorError;
use crate::simd;
use crate::DType;
use crate::Device;
use rand::Rng;
use rayon::prelude::*;

#[derive(Clone)]
pub struct Tensor {
    data: Vec<u8>,
    shape: Vec<usize>,
    dtype: DType,
    device: Device,
}

impl Tensor {
    /// Create a new tensor with the given shape and dtype, zero-filled.
    pub fn new(shape: &[usize], dtype: DType) -> Self {
        let num = shape.iter().product::<usize>();
        let bytes = (num * dtype.bit_width()).div_ceil(8);
        Self {
            data: vec![0u8; bytes],
            shape: shape.to_vec(),
            dtype,
            device: Device::Cpu,
        }
    }

    /// Create a f32 tensor from a slice.
    pub fn from_slice(data: &[f32], shape: &[usize]) -> Self {
        let num = shape.iter().product::<usize>();
        assert_eq!(data.len(), num);
        let mut t = Self::new(shape, DType::F32);
        let mut raw = Vec::with_capacity(num * 4);
        for v in data {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        t.data = raw;
        t
    }

    pub fn random(shape: &[usize], dtype: DType) -> Self {
        let num = shape.iter().product::<usize>();
        let rng = rand::thread_rng();
        let raw: Vec<f32> = rng
            .sample_iter(rand::distributions::Uniform::new(-1.0, 1.0))
            .take(num)
            .collect();
        let mut t = Self::from_slice(&raw, shape);
        if dtype != DType::F32 {
            t = t.to_dtype(dtype);
        }
        t
    }

    pub fn on_device(shape: &[usize], dtype: DType, device: Device) -> Self {
        let mut t = Self::new(shape, dtype);
        t.device = device;
        t
    }

    pub fn zeros(shape: &[usize], dtype: DType) -> Self {
        Self::new(shape, dtype)
    }

    pub fn ones(shape: &[usize], dtype: DType) -> Self {
        let mut t = Self::new(shape, DType::F32);
        let n = t.num_elements();
        for i in 0..n {
            t.set_flat_f32(i, 1.0);
        }
        if dtype != DType::F32 {
            t = t.to_dtype(dtype);
        }
        t
    }

    // accessors

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    pub fn device(&self) -> Device {
        self.device
    }
    pub fn is_gpu(&self) -> bool {
        matches!(self.device, Device::Gpu { .. })
    }
    pub fn data(&self) -> &[u8] {
        &self.data
    }
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }
    pub fn nbytes(&self) -> usize {
        self.data.len()
    }
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    // shape manipulation

    pub fn reshape(&self, new_shape: &[usize]) -> Self {
        let new_num = new_shape.iter().product::<usize>();
        assert_eq!(new_num, self.num_elements());
        Self {
            data: self.data.clone(),
            shape: new_shape.to_vec(),
            dtype: self.dtype,
            device: self.device,
        }
    }

    pub fn transpose(&self) -> Self {
        assert_eq!(self.ndim(), 2);
        let rows = self.shape[0];
        let cols = self.shape[1];
        let mut result = Self::new(&[cols, rows], self.dtype);
        let left = self.to_f32();
        for r in 0..rows {
            for c in 0..cols {
                result.set_flat_f32(c * rows + r, left.get_flat_f32(r * cols + c));
            }
        }
        if self.dtype != DType::F32 {
            result = result.to_dtype(self.dtype);
        }
        result
    }

    pub fn flatten(&self) -> Self {
        Self {
            data: self.data.clone(),
            shape: vec![self.num_elements()],
            dtype: self.dtype,
            device: self.device,
        }
    }

    // value get/set

    pub fn get_flat_f32(&self, idx: usize) -> f32 {
        match self.dtype {
            DType::F32 => {
                let base = idx * 4;
                f32::from_le_bytes(self.data[base..base + 4].try_into().unwrap())
            }
            DType::F16 => f16_to_f32(self.u16_at(idx)),
            DType::BF16 => bf16_to_f32(self.u16_at(idx)),
            DType::INT8 => self.data[idx] as i8 as f32 / 127.0,
            DType::INT4 => {
                let half = (self.data[idx / 2] >> (4 * (idx % 2))) & 0x0f;
                let val = if half & 0x08 != 0 {
                    half as i8 - 16
                } else {
                    half as i8
                };
                (val as f32) / 7.0
            }
            DType::BIT1 => {
                if (self.data[idx / 8] >> (idx % 8)) & 1 == 1 {
                    1.0
                } else {
                    -1.0
                }
            }
        }
    }

    pub fn set_flat_f32(&mut self, idx: usize, val: f32) {
        match self.dtype {
            DType::F32 => {
                self.data[idx * 4..idx * 4 + 4].copy_from_slice(&val.to_le_bytes());
            }
            DType::F16 => {
                self.data[idx * 2..idx * 2 + 2].copy_from_slice(&f32_to_f16(val).to_le_bytes())
            }
            DType::BF16 => {
                self.data[idx * 2..idx * 2 + 2].copy_from_slice(&f32_to_bf16(val).to_le_bytes())
            }
            DType::INT8 => self.data[idx] = (val.clamp(-1.0, 1.0) * 127.0) as i8 as u8,
            DType::INT4 => {
                let v = (val.clamp(-1.0, 1.0) * 7.0) as i8 & 0x0f;
                let shift = 4 * (idx % 2);
                self.data[idx / 2] = (self.data[idx / 2] & !(0x0f << shift)) | ((v as u8) << shift);
            }
            DType::BIT1 => {
                let byte = idx / 8;
                let bit = idx % 8;
                if val > 0.0 {
                    self.data[byte] |= 1 << bit;
                } else {
                    self.data[byte] &= !(1 << bit);
                }
            }
        }
    }

    pub fn get_f32(&self, indices: &[usize]) -> f32 {
        assert_eq!(indices.len(), self.ndim());
        let mut idx = 0;
        for (i, &dim) in self.shape.iter().enumerate() {
            idx = idx * dim + indices[i];
        }
        self.get_flat_f32(idx)
    }

    pub fn set_f32(&mut self, indices: &[usize], val: f32) {
        assert_eq!(indices.len(), self.ndim());
        let mut idx = 0;
        for (i, &dim) in self.shape.iter().enumerate() {
            idx = idx * dim + indices[i];
        }
        self.set_flat_f32(idx, val);
    }

    // dtype conversion

    pub fn to_dtype(&self, target: DType) -> Self {
        if target == DType::F32 {
            return self.to_f32();
        }
        if target == DType::F16 {
            return self.to_f16();
        }
        if target == DType::BF16 {
            return self.to_bf16();
        }
        if target == DType::INT8 {
            return self.to_int8();
        }
        if target == DType::INT4 {
            return self.to_int4();
        }
        if target == DType::BIT1 {
            return self.to_bit1();
        }
        unreachable!()
    }

    pub fn to_f32(&self) -> Self {
        let n = self.num_elements();
        let mut result = Self::new(&self.shape, DType::F32);
        for i in 0..n {
            result.set_flat_f32(i, self.get_flat_f32(i));
        }
        result
    }

    pub fn to_f16(&self) -> Self {
        // ok
        let src = self.to_f32();
        let mut result = Self::new(&self.shape, DType::F16);
        for i in 0..src.num_elements() {
            result.set_flat_f32(i, src.get_flat_f32(i));
        }
        result
    }

    pub fn to_bf16(&self) -> Self {
        // ok
        let src = self.to_f32();
        let mut result = Self::new(&self.shape, DType::BF16);
        for i in 0..src.num_elements() {
            result.set_flat_f32(i, src.get_flat_f32(i));
        }
        result
    }

    pub fn to_int8(&self) -> Self {
        let src = self.to_f32();
        let n = src.num_elements();
        let mut result = Self::new(&self.shape, DType::INT8);
        for i in 0..n {
            result.set_flat_f32(i, src.get_flat_f32(i));
        }
        result
    }

    pub fn to_int4(&self) -> Self {
        let src = self.to_f32();
        let n = src.num_elements();
        let mut result = Self::new(&self.shape, DType::INT4);
        for i in 0..n {
            let v = (src.get_flat_f32(i).clamp(-1.0, 1.0) * 7.0) as i8;
            if i % 2 == 0 {
                result.data[i / 2] &= 0xf0;
                result.data[i / 2] |= (v as u8) & 0x0f;
            } else {
                result.data[i / 2] &= 0x0f;
                result.data[i / 2] |= ((v as u8) & 0x0f) << 4;
            }
        }
        result
    }

    pub fn to_bit1(&self) -> Self {
        let src = self.to_f32();
        let n = src.num_elements();
        let packed = n.div_ceil(8);
        let mut result = Self::new(&self.shape, DType::BIT1);
        for i in 0..packed {
            let mut byte = 0u8;
            for b in 0..8 {
                let idx = i * 8 + b;
                if idx < n && src.get_flat_f32(idx) > 0.0 {
                    byte |= 1 << b;
                }
            }
            result.data[i] = byte;
        }
        result
    }

    // arithmetic

    pub fn f32_sum(&self) -> f32 {
        assert_eq!(self.dtype, DType::F32);
        simd::f32_sum(self.as_f32_slice())
    }

    pub fn f32_max(&self) -> f32 {
        assert_eq!(self.dtype, DType::F32);
        simd::f32_max(self.as_f32_slice())
    }

    pub fn f32_scale_inplace(&mut self, scale: f32) {
        assert_eq!(self.dtype, DType::F32);
        let out = self.as_f32_slice_mut();
        let tmp = out.to_vec();
        simd::f32_scale(&tmp, scale, out);
    }

    pub fn add(&self, other: &Self) -> Result<Self, TensorError> {
        if self.shape != other.shape {
            return Err(TensorError::ShapeMismatch("shape mismatch in add".into()));
        }
        let left = self.to_f32();
        let right = other.to_f32();
        let mut result = Self::new(&self.shape, DType::F32);
        {
            let a = left.as_f32_slice();
            let b = right.as_f32_slice();
            let out = result.as_f32_slice_mut();
            simd::f32_add(a, b, out);
        }
        Ok(result)
    }

    pub fn sub(&self, other: &Self) -> Result<Self, TensorError> {
        if self.shape != other.shape {
            return Err(TensorError::ShapeMismatch("shape mismatch in sub".into()));
        }
        let left = self.to_f32();
        let right = other.to_f32();
        let mut result = Self::new(&self.shape, DType::F32);
        {
            let a = left.as_f32_slice();
            let b = right.as_f32_slice();
            let out = result.as_f32_slice_mut();
            simd::f32_sub(a, b, out);
        }
        Ok(result)
    }

    pub fn mul(&self, other: &Self) -> Result<Self, TensorError> {
        if self.shape != other.shape {
            return Err(TensorError::ShapeMismatch("shape mismatch in mul".into()));
        }
        let left = self.to_f32();
        let right = other.to_f32();
        let mut result = Self::new(&self.shape, DType::F32);
        {
            let a = left.as_f32_slice();
            let b = right.as_f32_slice();
            let out = result.as_f32_slice_mut();
            simd::f32_mul(a, b, out);
        }
        Ok(result)
    }

    pub fn dot(&self, rhs: &Self) -> Result<Self, TensorError> {
        if self.ndim() != 2 || rhs.ndim() != 2 {
            return Err(TensorError::ShapeMismatch(
                "dot requires 2d matrices".into(),
            ));
        }
        if self.shape[1] != rhs.shape[0] {
            return Err(TensorError::ShapeMismatch(format!(
                "dot shape mismatch: {:?} x {:?}",
                self.shape, rhs.shape
            )));
        }
        let m = self.shape[0];
        let k = self.shape[1];
        let n = rhs.shape[1];
        let left = self.to_f32();
        let right = rhs.to_f32();

        let mut result = Self::new(&[m, n], DType::F32);

        let right_t = right.transpose();

        let a_slice = left.as_f32_slice();
        let bt_slice = right_t.as_f32_slice();
        let out_slice = result.as_f32_slice_mut();

        use rayon::slice::ParallelSliceMut;
        out_slice
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(i, out_row)| {
                simd::f32_matmul_row(&a_slice[i * k..(i + 1) * k], bt_slice, out_row, k, n);
            });

        Ok(result)
    }

    /// Get a contiguous f32 slice view. For F32 dtype only.
    /// Panics if dtype is not F32.
    pub fn as_f32_slice(&self) -> &[f32] {
        assert_eq!(self.dtype, DType::F32, "as_f32_slice requires F32 dtype");
        // SAFETY: we trust the layout is Vec<u8> of aligned f32s
        unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const f32, self.data.len() / 4) }
    }

    pub fn as_f32_slice_mut(&mut self) -> &mut [f32] {
        assert_eq!(
            self.dtype,
            DType::F32,
            "as_f32_slice_mut requires F32 dtype"
        );
        unsafe {
            std::slice::from_raw_parts_mut(self.data.as_mut_ptr() as *mut f32, self.data.len() / 4)
        }
    }

    fn u16_at(&self, idx: usize) -> u16 {
        u16::from_le_bytes(self.data[idx * 2..idx * 2 + 2].try_into().unwrap())
    }
}

// standalone helpers exported for use by other crates

pub fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign: u32 = (bits >> 16) & 0x8000;
    let exponent: i32 = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = bits & 0x7fffff;
    if exponent >= 30 {
        return (sign | 0x7c00) as u16;
    }
    if exponent <= 0 {
        return sign as u16;
    }
    let f16_bits = sign | ((exponent as u32) << 10) | (mantissa >> 13);
    f16_bits as u16
}

pub fn f16_to_f32(val: u16) -> f32 {
    let sign = ((val >> 15) & 1) as u32;
    let exponent = ((val >> 10) & 0x1f) as u32;
    let mantissa = (val & 0x3ff) as u32;
    if exponent == 0 && mantissa == 0 {
        return f32::from_bits(sign << 31);
    }
    if exponent == 0 {
        return f32::from_bits((sign << 31) | ((127 - 15) << 23) | (mantissa << 13));
    }
    if exponent == 31 {
        if mantissa == 0 {
            return f32::from_bits((sign << 31) | 0x7f800000);
        } else {
            return f32::from_bits((sign << 31) | 0x7fc00000);
        }
    }
    f32::from_bits((sign << 31) | ((exponent + 112) << 23) | (mantissa << 13))
}

pub fn f32_to_bf16(val: f32) -> u16 {
    let bits = val.to_bits();
    let rounding = ((bits >> 16) & 1) + 0x7fff;
    ((bits + rounding) >> 16) as u16
}

pub fn bf16_to_f32(val: u16) -> f32 {
    f32::from_bits((val as u32) << 16)
}
