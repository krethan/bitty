use crate::tensor::Tensor;

pub struct TensorView<'a> {
    tensor: &'a Tensor,
    offset: usize,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl<'a> TensorView<'a> {
    pub fn new(tensor: &'a Tensor, offset: usize, shape: Vec<usize>, strides: Vec<usize>) -> Self {
        Self {
            tensor,
            offset,
            shape,
            strides,
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    fn compute_index(&self, indices: &[usize]) -> usize {
        indices.iter().zip(&self.strides).map(|(i, s)| i * s).sum()
    }

    pub fn get_f32(&self, indices: &[usize]) -> f32 {
        let flat_idx = self.offset + self.compute_index(indices);
        self.tensor.get_flat_f32(flat_idx)
    }

    pub fn get_flat_f32(&self, idx: usize) -> f32 {
        let flat_idx = self.offset + self.compute_flat_index(idx);
        self.tensor.get_flat_f32(flat_idx)
    }

    fn compute_flat_index(&self, mut idx: usize) -> usize {
        let mut result = 0;
        for i in 0..self.shape.len() {
            let dim = self.shape[i];
            let stride_idx = idx / dim;
            let remainder = idx % dim;
            result += remainder * self.strides[i];
            idx = stride_idx;
        }
        result
    }

    pub fn transpose_view(&self) -> Self {
        assert_eq!(self.shape.len(), 2, "transpose_view only supports 2d tensors");
        let rows = self.shape[0];
        let cols = self.shape[1];
        Self {
            tensor: self.tensor,
            offset: self.offset,
            shape: vec![cols, rows],
            strides: vec![self.strides[1], self.strides[0]],
        }
    }
}
