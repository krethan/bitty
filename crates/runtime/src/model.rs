use crate::attention::{Attention, KvCache};
use crate::config::ModelConfig;
use crate::layers::{Embedding, Linear, RmsNorm};
use crate::sampler::Sampler;
use crate::GpuContext;
use bitllm_tensor::{DType, Tensor};

pub struct TransformerLayer {
    pub attention: Attention,
    pub attn_norm: RmsNorm,
    pub ffn_up: Linear,
    pub ffn_gate: Linear,
    pub ffn_down: Linear,
    pub ffn_norm: RmsNorm,
    pub config: ModelConfig,
}

impl TransformerLayer {
    pub fn forward(
        &self,
        input: &Tensor,
        cache: Option<&mut KvCache>,
        layer_idx: usize,
        position: usize,
    ) -> Tensor {
        self.forward_gpu(input, cache, layer_idx, position, None)
    }

    pub fn forward_gpu(
        &self,
        input: &Tensor,
        cache: Option<&mut KvCache>,
        layer_idx: usize,
        position: usize,
        gpu: Option<&GpuContext>,
    ) -> Tensor {
        let residual = input.clone();
        let normed = self.attn_norm.forward_gpu(input, gpu);
        let attn_out = self
            .attention
            .forward_gpu(&normed, cache, layer_idx, position, gpu);
        let h = gpu_add(&residual, &attn_out, gpu);

        let residual2 = h.clone();
        let normed2 = self.ffn_norm.forward_gpu(&h, gpu);
        let up = self.ffn_up.forward_gpu(&normed2, gpu);
        let gate = self.ffn_gate.forward_gpu(&normed2, gpu);
        let gated = silu(&gate);
        let activated = hadamard(&gated, &up);
        let ffn_out = self.ffn_down.forward_gpu(&activated, gpu);

        gpu_add(&residual2, &ffn_out, gpu)
    }
}

pub struct Model {
    pub config: ModelConfig,
    pub embedding: Embedding,
    pub layers: Vec<TransformerLayer>,
    pub norm: RmsNorm,
    pub lm_head: Linear,
    pub cache: Option<KvCache>,
    #[cfg(feature = "gpu")]
    pub gpu: Option<GpuContext>,
}

impl Model {
    pub fn new(config: ModelConfig) -> Self {
        let embedding = Embedding::new(
            Tensor::zeros(&[config.vocab_size, config.hidden_size], DType::F32),
            config.vocab_size,
            config.hidden_size,
        );

        let norm = RmsNorm::new(
            Tensor::ones(&[config.hidden_size], DType::F32),
            config.norm_eps,
        );

        let lm_head = Linear::new(
            Tensor::zeros(&[config.vocab_size, config.hidden_size], DType::F32),
            None,
        );

        let layers = (0..config.num_layers)
            .map(|_| create_dummy_layer(&config))
            .collect();

        let cache = Some(KvCache::new(
            config.num_layers,
            config.max_seq_len,
            config.num_kv_heads(),
            config.head_dim(),
        ));

        Self {
            config,
            embedding,
            layers,
            norm,
            lm_head,
            cache,
            #[cfg(feature = "gpu")]
            gpu: None,
        }
    }

    #[cfg(feature = "gpu")]
    pub fn set_gpu(&mut self, ctx: GpuContext) {
        self.gpu = Some(ctx);
    }

    pub fn forward(&mut self, token_ids: &[u32]) -> Tensor {
        self.forward_gpu(token_ids, None)
    }

    pub fn forward_gpu(&mut self, token_ids: &[u32], gpu: Option<&GpuContext>) -> Tensor {
        let seq_len = token_ids.len();
        let pos = self.cache.as_ref().map_or(0, |c| c.get_seq_len());

        let mut hidden = self.embedding.forward(token_ids);

        if let Some(ref mut cache) = self.cache {
            for (i, layer) in self.layers.iter().enumerate() {
                hidden = layer.forward_gpu(&hidden, Some(cache), i, pos + i, gpu);
            }
            cache.advance(seq_len);
        } else {
            for (i, layer) in self.layers.iter().enumerate() {
                hidden = layer.forward_gpu(&hidden, None, i, pos, gpu);
            }
        }

        let normed = self.norm.forward_gpu(&hidden, gpu);
        self.lm_head.forward_gpu(&normed, gpu)
    }

    pub fn generate(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        sampler: &Sampler,
    ) -> Vec<u32> {
        let mut tokens = prompt_tokens.to_vec();
        let mut generated = Vec::new();

        #[cfg(feature = "gpu")]
        let gpu_ctx = self.gpu.clone();
        #[cfg(not(feature = "gpu"))]
        let gpu_ctx: Option<GpuContext> = None;

        let logits = self.forward_gpu(&tokens, gpu_ctx.as_ref());
        let last_logits_row = logits.shape()[0] - 1;
        let vocab_size = self.config.vocab_size;
        let mut logits_vec = Vec::with_capacity(vocab_size);
        for j in 0..vocab_size {
            logits_vec.push(logits.get_flat_f32(last_logits_row * vocab_size + j));
        }
        let next_token = sampler.sample(&logits_vec);
        tokens.push(next_token);
        generated.push(next_token);

        let mut next_token = next_token;

        for _ in 1..max_new_tokens {
            let logits = self.forward_gpu(&[next_token], gpu_ctx.as_ref());
            let vocab_size = self.config.vocab_size;
            let mut logits_vec = Vec::with_capacity(vocab_size);
            for j in 0..vocab_size {
                logits_vec.push(logits.get_flat_f32(j));
            }
            next_token = sampler.sample(&logits_vec);
            tokens.push(next_token);
            generated.push(next_token);
        }

        generated
    }

    pub fn clear_cache(&mut self) {
        if let Some(ref mut cache) = self.cache {
            *cache = KvCache::new(
                self.config.num_layers,
                self.config.max_seq_len,
                self.config.num_kv_heads(),
                self.config.head_dim(),
            );
        }
    }

    pub fn generate_streaming(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        sampler: &Sampler,
        tx: tokio::sync::mpsc::Sender<u32>,
    ) {
        let mut tokens = prompt_tokens.to_vec();

        #[cfg(feature = "gpu")]
        let gpu_ctx = self.gpu.clone();
        #[cfg(not(feature = "gpu"))]
        let gpu_ctx: Option<GpuContext> = None;

        let logits = self.forward_gpu(&tokens, gpu_ctx.as_ref());
        let last_logits_row = logits.shape()[0] - 1;
        let vocab_size = self.config.vocab_size;
        let mut logits_vec = Vec::with_capacity(vocab_size);
        for j in 0..vocab_size {
            logits_vec.push(logits.get_flat_f32(last_logits_row * vocab_size + j));
        }
        let next_token = sampler.sample(&logits_vec);
        tokens.push(next_token);
        let _ = tx.blocking_send(next_token);

        let mut next_token = next_token;

        for _ in 1..max_new_tokens {
            let logits = self.forward_gpu(&[next_token], gpu_ctx.as_ref());
            let vocab_size = self.config.vocab_size;
            let mut logits_vec = Vec::with_capacity(vocab_size);
            for j in 0..vocab_size {
                logits_vec.push(logits.get_flat_f32(j));
            }
            next_token = sampler.sample(&logits_vec);
            tokens.push(next_token);
            if tx.blocking_send(next_token).is_err() {
                break;
            }
        }
    }
}

fn create_dummy_layer(config: &ModelConfig) -> TransformerLayer {
    let hidden = config.hidden_size;
    let head_dim = config.head_dim();

    TransformerLayer {
        attention: Attention::new(
            Linear::new(Tensor::zeros(&[hidden, hidden], DType::F32), None),
            Linear::new(
                Tensor::zeros(&[hidden, config.num_kv_heads() * head_dim], DType::F32),
                None,
            ),
            Linear::new(
                Tensor::zeros(&[hidden, config.num_kv_heads() * head_dim], DType::F32),
                None,
            ),
            Linear::new(Tensor::zeros(&[hidden, hidden], DType::F32), None),
            config.clone(),
        ),
        attn_norm: RmsNorm::new(Tensor::ones(&[hidden], DType::F32), config.norm_eps),
        ffn_up: Linear::new(
            Tensor::zeros(&[config.intermediate_size, hidden], DType::F32),
            None,
        ),
        ffn_gate: Linear::new(
            Tensor::zeros(&[config.intermediate_size, hidden], DType::F32),
            None,
        ),
        ffn_down: Linear::new(
            Tensor::zeros(&[hidden, config.intermediate_size], DType::F32),
            None,
        ),
        ffn_norm: RmsNorm::new(Tensor::ones(&[hidden], DType::F32), config.norm_eps),
        config: config.clone(),
    }
}

fn silu(x: &Tensor) -> Tensor {
    let n = x.num_elements();
    let mut result = Tensor::zeros(x.shape(), DType::F32);
    for i in 0..n {
        let v = x.get_flat_f32(i);
        let s = 1.0 / (1.0 + (-v).exp());
        result.set_flat_f32(i, v * s);
    }
    result
}

fn hadamard(a: &Tensor, b: &Tensor) -> Tensor {
    let n = a.num_elements();
    let mut result = Tensor::zeros(a.shape(), DType::F32);
    for i in 0..n {
        result.set_flat_f32(i, a.get_flat_f32(i) * b.get_flat_f32(i));
    }
    result
}

fn gpu_add(a: &Tensor, b: &Tensor, gpu: Option<&GpuContext>) -> Tensor {
    #[cfg(feature = "gpu")]
    if let Some(ctx) = gpu {
        return ctx.add(a, b).unwrap_or_else(|e| {
            log::warn!("GPU add failed, falling back to CPU: {}", e);
            a.add(b).unwrap()
        });
    }
    let _ = gpu;
    a.add(b).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_creation() {
        let config = ModelConfig::tiny_test();
        let model = Model::new(config.clone());
        assert_eq!(model.config.vocab_size, config.vocab_size);
        assert_eq!(model.layers.len(), config.num_layers);
    }

    #[test]
    fn test_model_forward() {
        let config = ModelConfig::tiny_test();
        let mut model = Model::new(config);
        let tokens = vec![0u32, 1, 2];
        let logits = model.forward(&tokens);
        assert_eq!(logits.shape(), &[3, 256]);
    }

    #[test]
    fn test_silu() {
        let t = Tensor::from_slice(&[0.0, 1.0, -1.0], &[3]);
        let result = silu(&t);
        assert!((result.get_flat_f32(0) - 0.0).abs() < 1e-6);
        assert!((result.get_flat_f32(1) - (1.0 / (1.0 + (-1.0f32).exp()))).abs() < 1e-6);
    }

    #[test]
    fn test_generation() {
        let config = ModelConfig::tiny_test();
        let mut model = Model::new(config);
        let sampler = Sampler::greedy();
        let generated = model.generate(&[0, 1], 5, &sampler);
        assert_eq!(generated.len(), 5);
    }
}
