use bitllm_tensor::{DType, Tensor};
use std::sync::Arc;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum DistributedError {
    #[error("Tensor parallelism requires at least 2 devices")]
    InsufficientDevices,
    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),
    #[error("Communication error: {0}")]
    CommunicationError(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelStrategy {
    Tensor,
    Pipeline,
}

pub struct DeviceMesh {
    pub world_size: usize,
    pub rank: usize,
}

impl DeviceMesh {
    pub fn new(world_size: usize, rank: usize) -> Result<Self, DistributedError> {
        if world_size < 2 {
            return Err(DistributedError::InsufficientDevices);
        }
        if rank >= world_size {
            return Err(DistributedError::ShapeMismatch(format!(
                "rank {} >= world_size {}",
                rank, world_size
            )));
        }
        Ok(Self { world_size, rank })
    }

    pub fn local() -> Self {
        Self {
            world_size: 1,
            rank: 0,
        }
    }

    pub fn is_main(&self) -> bool {
        self.rank == 0
    }
}

pub struct TensorPartitioner;

impl TensorPartitioner {
    pub fn partition_along_dim(
        tensor: &Tensor,
        dim: usize,
        rank: usize,
        world_size: usize,
    ) -> Tensor {
        let shape = tensor.shape();
        assert!(dim < shape.len(), "dim {} >= ndim {}", dim, shape.len());

        let dim_size = shape[dim];
        let chunk_size = dim_size.div_ceil(world_size);
        let start = (rank * chunk_size).min(dim_size);
        let end = ((rank + 1) * chunk_size).min(dim_size);
        let local_size = end - start;

        let mut new_shape = shape.to_vec();
        new_shape[dim] = local_size;

        let mut result = Tensor::new(&new_shape, tensor.dtype());
        let src_shape = shape;
        let dst_shape = &new_shape;

        let outer: usize = dst_shape[..dim].iter().product();
        let inner: usize = dst_shape[dim + 1..].iter().product();

        for o in 0..outer {
            for i in 0..local_size {
                for j in 0..inner {
                    let src_flat = Self::compute_flat_index(
                        src_shape,
                        &Self::compose_index(o, dim, i + start, j, src_shape),
                    );
                    let dst_flat = Self::compute_flat_index(
                        dst_shape,
                        &Self::compose_index(o, dim, i, j, dst_shape),
                    );
                    let val = tensor.get_flat_f32(src_flat);
                    result.set_flat_f32(dst_flat, val);
                }
            }
        }

        result
    }

    pub fn all_gather(partitions: &[Tensor]) -> Tensor {
        assert!(!partitions.is_empty(), "no partitions to gather");
        if partitions.len() == 1 {
            return partitions[0].clone();
        }

        let ndim = partitions[0].ndim();
        let concat_dim = ndim - 1;
        let mut total_size = 0;
        for p in partitions {
            total_size += p.shape()[concat_dim];
        }

        let mut new_shape = partitions[0].shape().to_vec();
        new_shape[concat_dim] = total_size;

        let mut result = Tensor::new(&new_shape, partitions[0].dtype());
        let mut offset = 0;

        for p in partitions {
            let chunk_size = p.shape()[concat_dim];
            let outer: usize = p.shape()[..concat_dim].iter().product();
            let inner: usize = if concat_dim + 1 < p.ndim() {
                p.shape()[concat_dim + 1..].iter().product()
            } else {
                1
            };

            for o in 0..outer {
                for i in 0..chunk_size {
                    for j in 0..inner {
                        let src_idx = Self::compute_flat_index(
                            p.shape(),
                            &Self::compose_index(o, concat_dim, i, j, p.shape()),
                        );
                        let dst_idx = Self::compute_flat_index(
                            &new_shape,
                            &Self::compose_index(o, concat_dim, offset + i, j, &new_shape),
                        );
                        result.set_flat_f32(dst_idx, p.get_flat_f32(src_idx));
                    }
                }
            }
            offset += chunk_size;
        }

        result
    }

    pub fn reduce_sum(partitions: &[Tensor]) -> Tensor {
        assert!(!partitions.is_empty());
        let mut result = partitions[0].clone();
        for p in &partitions[1..] {
            result = result.add(p).unwrap();
        }
        result
    }

    fn compute_flat_index(shape: &[usize], indices: &[usize]) -> usize {
        let mut flat = 0;
        let mut stride = 1;
        for i in (0..shape.len()).rev() {
            flat += indices[i] * stride;
            stride *= shape[i];
        }
        flat
    }

    fn compose_index(
        outer: usize,
        dim: usize,
        dim_val: usize,
        inner: usize,
        shape: &[usize],
    ) -> Vec<usize> {
        let ndim = shape.len();
        let mut indices = vec![0; ndim];

        let mut remaining = outer;
        for i in (0..dim).rev() {
            indices[i] = remaining % shape[i];
            remaining /= shape[i];
        }

        indices[dim] = dim_val;

        remaining = inner;
        for i in (dim + 1..ndim).rev() {
            indices[i] = remaining % shape[i];
            remaining /= shape[i];
        }

        indices
    }
}

pub struct TensorParallelLinear {
    partitions: Vec<Arc<Tensor>>,
    mesh: DeviceMesh,
}

impl TensorParallelLinear {
    pub fn new(weight: &Tensor, mesh: DeviceMesh, parallel_dim: usize) -> Self {
        let partitions: Vec<Arc<Tensor>> = (0..mesh.world_size)
            .map(|rank| {
                let partition = TensorPartitioner::partition_along_dim(
                    weight,
                    parallel_dim,
                    rank,
                    mesh.world_size,
                );
                Arc::new(partition)
            })
            .collect();
        Self { partitions, mesh }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        let local_weight = &self.partitions[self.mesh.rank];
        let m = input.shape()[0];
        let k = input.shape()[1];
        let n = local_weight.shape()[0];
        let mut result = Tensor::zeros(&[m, n], DType::F32);
        {
            let a = input.as_f32_slice();
            let b = local_weight.as_f32_slice();
            let out = result.as_f32_slice_mut();
            bitllm_tensor::simd::f32_matmul(a, b, out, m, k, n);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_mesh() {
        let mesh = DeviceMesh::new(4, 2).unwrap();
        assert_eq!(mesh.world_size, 4);
        assert_eq!(mesh.rank, 2);
        assert!(!mesh.is_main());
    }

    #[test]
    fn test_local_mesh() {
        let mesh = DeviceMesh::local();
        assert!(mesh.is_main());
        assert_eq!(mesh.world_size, 1);
    }

    #[test]
    fn test_partition_along_last_dim() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let p0 = TensorPartitioner::partition_along_dim(&t, 1, 0, 2);
        let p1 = TensorPartitioner::partition_along_dim(&t, 1, 1, 2);

        assert_eq!(p0.shape(), &[2, 2]);
        assert_eq!(p1.shape(), &[2, 1]);

        assert_eq!(p0.get_flat_f32(0), 1.0);
        assert_eq!(p0.get_flat_f32(1), 2.0);
        assert_eq!(p0.get_flat_f32(2), 4.0);
        assert_eq!(p0.get_flat_f32(3), 5.0);

        assert_eq!(p1.get_flat_f32(0), 3.0);
        assert_eq!(p1.get_flat_f32(1), 6.0);
    }

    #[test]
    fn test_all_gather() {
        let p0 = Tensor::from_slice(&[1.0, 2.0], &[2]);
        let p1 = Tensor::from_slice(&[3.0, 4.0], &[2]);
        let gathered = TensorPartitioner::all_gather(&[p0, p1]);
        assert_eq!(gathered.shape(), &[4]);
        assert_eq!(gathered.get_flat_f32(0), 1.0);
        assert_eq!(gathered.get_flat_f32(3), 4.0);
    }

    #[test]
    fn test_reduce_sum() {
        let p0 = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let p1 = Tensor::from_slice(&[4.0, 5.0, 6.0], &[3]);
        let result = TensorPartitioner::reduce_sum(&[p0, p1]);
        assert_eq!(result.get_flat_f32(0), 5.0);
        assert_eq!(result.get_flat_f32(1), 7.0);
        assert_eq!(result.get_flat_f32(2), 9.0);
    }

    #[test]
    fn test_partition_and_gather_roundtrip() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]);
        let p0 = TensorPartitioner::partition_along_dim(&t, 1, 0, 2);
        let p1 = TensorPartitioner::partition_along_dim(&t, 1, 1, 2);
        let gathered = TensorPartitioner::all_gather(&[p0, p1]);

        assert_eq!(gathered.shape(), t.shape());
        for i in 0..8 {
            assert_eq!(gathered.get_flat_f32(i), t.get_flat_f32(i));
        }
    }

    #[test]
    fn test_partition_three_way() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3]);
        let p0 = TensorPartitioner::partition_along_dim(&t, 1, 0, 3);
        let p1 = TensorPartitioner::partition_along_dim(&t, 1, 1, 3);
        let p2 = TensorPartitioner::partition_along_dim(&t, 1, 2, 3);

        assert_eq!(p0.shape(), &[3, 1]);
        assert_eq!(p1.shape(), &[3, 1]);
        assert_eq!(p2.shape(), &[3, 1]);

        let gathered = TensorPartitioner::all_gather(&[p0, p1, p2]);
        assert_eq!(gathered.shape(), &[3, 3]);
        for i in 0..9 {
            assert_eq!(gathered.get_flat_f32(i), t.get_flat_f32(i));
        }
    }

    #[test]
    fn test_compose_index() {
        let shape = vec![2, 3, 4];
        let idx = TensorPartitioner::compose_index(2, 1, 1, 3, &shape);
        assert_eq!(idx, vec![0, 1, 3]);
    }
}
