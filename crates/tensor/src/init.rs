use crate::tensor::Tensor;
use rand::Rng;

pub enum InitMethod {
    XavierNormal,
    XavierUniform,
    HeNormal,
    HeUniform,
    Constant(f32),
}

impl InitMethod {
    pub fn initialize(&self, tensor: &mut Tensor) {
        let n = tensor.num_elements();
        let shape = tensor.shape().to_vec();
        let fan_in = if shape.len() >= 2 {
            shape[shape.len() - 2]
        } else {
            1
        };
        let fan_out = shape[shape.len() - 1];
        let mut rng = rand::thread_rng();
        match self {
            InitMethod::XavierNormal | InitMethod::HeNormal => {
                let std = if matches!(self, InitMethod::XavierNormal) {
                    (6.0 / (fan_in + fan_out) as f32).sqrt()
                } else {
                    (2.0 / fan_in as f32).sqrt()
                };
                for i in 0..n {
                    let val: f32 = rng.gen_range(-std..std);
                    tensor.set_flat_f32(i, val);
                }
            }
            InitMethod::XavierUniform | InitMethod::HeUniform => {
                let limit = if matches!(self, InitMethod::XavierUniform) {
                    (6.0 / (fan_in + fan_out) as f32).sqrt()
                } else {
                    (6.0 / fan_in as f32).sqrt()
                };
                for i in 0..n {
                    let val: f32 = rng.gen_range(-limit..limit);
                    tensor.set_flat_f32(i, val);
                }
            }
            InitMethod::Constant(v) => {
                for i in 0..n {
                    tensor.set_flat_f32(i, *v);
                }
            }
        }
    }
}
