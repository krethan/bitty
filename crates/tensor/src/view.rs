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
}
