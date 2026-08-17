use crate::error::TensorError;
use crate::simd;
use crate::DType;
use crate::Device;
use rand::Rng;

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
        let bytes = num * 4;
        let mut raw = vec![0u8; bytes];
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, raw.as_mut_ptr(), bytes);
        }
        Self {
            data: raw,
            shape: shape.to_vec(),
            dtype: DType::F32,
            device: Device::Cpu,
        }
    }

    /// Try to create a f32 tensor from a slice, returning an error on mismatch.
    pub fn try_from_slice(data: &[f32], shape: &[usize]) -> Result<Self, TensorError> {
        let num = shape.iter().product::<usize>();
        if data.len() != num {
            return Err(TensorError::ShapeMismatch(format!(
                "from_slice: data len {} != shape product {}",
                data.len(),
                num
            )));
        }
        Ok(Self::from_slice(data, shape))
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
        let out = t.as_f32_slice_mut();
        for v in out.iter_mut() {
            *v = 1.0;
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

    /// Reinterpret the tensor's shape without cloning the data buffer.
    /// Panics on element count mismatch.
    pub fn reshape_owned(mut self, new_shape: &[usize]) -> Self {
        let new_num = new_shape.iter().product::<usize>();
        assert_eq!(new_num, self.num_elements());
        self.shape = new_shape.to_vec();
        self
    }

    pub fn try_reshape(&self, new_shape: &[usize]) -> Result<Self, TensorError> {
        let new_num = new_shape.iter().product::<usize>();
        if new_num != self.num_elements() {
            return Err(TensorError::ShapeMismatch(format!(
                "reshape: new shape product {} != current {}",
                new_num,
                self.num_elements()
            )));
        }
        Ok(Self {
            data: self.data.clone(),
            shape: new_shape.to_vec(),
            dtype: self.dtype,
            device: self.device,
        })
    }

    pub fn transpose(&self) -> Self {
        assert_eq!(self.ndim(), 2);
        let rows = self.shape[0];
        let cols = self.shape[1];
        let mut result = Self::new(&[cols, rows], self.dtype);

        // Fast path: F32 dtype with direct slice access. The naive
        // get_flat_f32/set_flat_f32 path does a dtype dispatch and byte-by-byte
        // copy on every element; pinning F32 and using direct slices is
        // ~8-16x faster on large matrices.
        if self.dtype == DType::F32 {
            let src = self.as_f32_slice();
            let dst = result.as_f32_slice_mut();
            for r in 0..rows {
                let src_row = &src[r * cols..(r + 1) * cols];
                for (c, &v) in src_row.iter().enumerate() {
                    dst[c * rows + r] = v;
                }
            }
        } else {
            let left = self.to_f32();
            let src = left.as_f32_slice();
            for r in 0..rows {
                let src_row = &src[r * cols..(r + 1) * cols];
                for (c, &v) in src_row.iter().enumerate() {
                    result.set_flat_f32(c * rows + r, v);
                }
            }
            if self.dtype != DType::F32 {
                result = result.to_dtype(self.dtype);
            }
        }

        result
    }

    /// Returns a lazy transposed view without allocating a new tensor.
    pub fn transpose_view(&self) -> crate::view::TensorView<'_> {
        assert_eq!(self.ndim(), 2, "transpose_view requires a 2d tensor");
        let rows = self.shape[0];
        let cols = self.shape[1];
        crate::view::TensorView::new(
            self,
            0,
            vec![cols, rows],
            vec![1, cols], // row-major: stride for dim0=cols, stride for dim1=1
        )
    }

    pub fn try_transpose(&self) -> Result<Self, TensorError> {
        if self.ndim() != 2 {
            return Err(TensorError::ShapeMismatch(format!(
                "transpose: expected 2d tensor, got {}d",
                self.ndim()
            )));
        }
        Ok(self.transpose())
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
        match target {
            DType::F32 => self.to_f32(),
            DType::BIT1 => self.to_bit1(),
        }
    }

    pub fn to_f32(&self) -> Self {
        if self.dtype == DType::F32 {
            return self.clone();
        }
        let n = self.num_elements();
        let mut result = Self::new(&self.shape, DType::F32);
        let out = result.as_f32_slice_mut();
        match self.dtype {
            DType::BIT1 => {
                for (i, o) in out.iter_mut().enumerate().take(n) {
                    *o = self.get_flat_f32(i);
                }
            }
            DType::F32 => unreachable!(),
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
        let mut result = Self::new(&self.shape, DType::F32);
        {
            let out = result.as_f32_slice_mut();
            if self.dtype == DType::F32 && other.dtype == DType::F32 {
                simd::f32_add(self.as_f32_slice(), other.as_f32_slice(), out);
            } else {
                let left = self.to_f32();
                let right = other.to_f32();
                simd::f32_add(left.as_f32_slice(), right.as_f32_slice(), out);
            }
        }
        Ok(result)
    }

    pub fn add_assign(&mut self, other: &Self) -> Result<(), TensorError> {
        if self.shape != other.shape {
            return Err(TensorError::ShapeMismatch("shape mismatch in add".into()));
        }
        if self.dtype == DType::F32 && other.dtype == DType::F32 {
            simd::f32_axpy(other.as_f32_slice(), 1.0, self.as_f32_slice_mut());
        } else {
            let self_f32 = self.to_f32();
            let other_f32 = other.to_f32();
            for ((o, a), b) in self
                .as_f32_slice_mut()
                .iter_mut()
                .zip(self_f32.as_f32_slice())
                .zip(other_f32.as_f32_slice())
            {
                *o = *a + b;
            }
        }
        Ok(())
    }

    pub fn sub(&self, other: &Self) -> Result<Self, TensorError> {
        if self.shape != other.shape {
            return Err(TensorError::ShapeMismatch("shape mismatch in sub".into()));
        }
        let mut result = Self::new(&self.shape, DType::F32);
        {
            let out = result.as_f32_slice_mut();
            if self.dtype == DType::F32 && other.dtype == DType::F32 {
                simd::f32_sub(self.as_f32_slice(), other.as_f32_slice(), out);
            } else {
                let left = self.to_f32();
                let right = other.to_f32();
                simd::f32_sub(left.as_f32_slice(), right.as_f32_slice(), out);
            }
        }
        Ok(result)
    }

    pub fn mul(&self, other: &Self) -> Result<Self, TensorError> {
        if self.shape != other.shape {
            return Err(TensorError::ShapeMismatch("shape mismatch in mul".into()));
        }
        let mut result = Self::new(&self.shape, DType::F32);
        {
            let out = result.as_f32_slice_mut();
            if self.dtype == DType::F32 && other.dtype == DType::F32 {
                simd::f32_mul(self.as_f32_slice(), other.as_f32_slice(), out);
            } else {
                let left = self.to_f32();
                let right = other.to_f32();
                simd::f32_mul(left.as_f32_slice(), right.as_f32_slice(), out);
            }
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

        let a_owned;
        let b_owned;
        let a_slice = if self.dtype == DType::F32 {
            self.as_f32_slice()
        } else {
            a_owned = self.to_f32();
            a_owned.as_f32_slice()
        };
        let b_slice = if rhs.dtype == DType::F32 {
            rhs.as_f32_slice()
        } else {
            b_owned = rhs.to_f32();
            b_owned.as_f32_slice()
        };

        let mut result = Self::new(&[m, n], DType::F32);
        let out_slice = result.as_f32_slice_mut();

        // Fast path: single-row input (m=1, common in inference for a single
        // token). Each output column is a dot product over k. The SIMD
        // f32_dot kernel handles this with cache-friendly contiguous access.
        if m == 1 {
            for j in 0..n {
                // b_slice[j * k .. (j+1) * k] is contiguous in row-major rhs.
                out_slice[j] = simd::f32_dot(a_slice, &b_slice[j * k..(j + 1) * k]);
            }
            return Ok(result);
        }

        // Multi-row path: transpose rhs in-place to match the SIMD kernel's
        // expected layout (b_t[i + j * k]), then dispatch to the SIMD matmul.
        // The transpose uses direct f32 slice access (8-16x faster than the
        // byte-by-byte Tensor::transpose that goes through get_flat_f32).
        let mut b_t = vec![0.0f32; k * n];
        for r in 0..k {
            for c in 0..n {
                b_t[r + c * k] = b_slice[r * n + c];
            }
        }
        simd::f32_matmul(a_slice, &b_t, out_slice, m, k, n);

        Ok(result)
    }

    /// Get a contiguous f32 slice view. For F32 dtype only.
    /// Panics if dtype is not F32.
    pub fn as_f32_slice(&self) -> &[f32] {
        assert_eq!(self.dtype, DType::F32, "as_f32_slice requires F32 dtype");
        assert_eq!(
            self.data.as_ptr() as usize % std::mem::align_of::<f32>(),
            0,
            "as_f32_slice requires f32 alignment"
        );
        unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const f32, self.data.len() / 4) }
    }

    pub fn as_f32_slice_mut(&mut self) -> &mut [f32] {
        assert_eq!(
            self.dtype,
            DType::F32,
            "as_f32_slice_mut requires F32 dtype"
        );
        assert_eq!(
            self.data.as_ptr() as usize % std::mem::align_of::<f32>(),
            0,
            "as_f32_slice_mut requires f32 alignment"
        );
        unsafe {
            std::slice::from_raw_parts_mut(self.data.as_mut_ptr() as *mut f32, self.data.len() / 4)
        }
    }
}
