use crate::attention::{Attention, KvCache, RoPECache};
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
        self.forward_gpu(input, cache, layer_idx, position, None, None)
    }

    pub fn forward_gpu(
        &self,
        input: &Tensor,
        cache: Option<&mut KvCache>,
        layer_idx: usize,
        position: usize,
        gpu: Option<&GpuContext>,
        rope_cache: Option<&RoPECache>,
    ) -> Tensor {
        let normed = self.attn_norm.forward_gpu(input, gpu);
        let attn_out = self
            .attention
            .forward_gpu_with_rope_cache(&normed, cache, layer_idx, position, gpu, rope_cache);

        #[cfg(feature = "gpu")]
        if let Some(ctx) = gpu {
            let h = ctx.add(input, &attn_out).unwrap_or_else(|e| {
                log::warn!("GPU add failed, falling back to CPU: {}", e);
                let mut h = input.clone();
                h.add_assign(&attn_out).unwrap();
                h
            });
            let normed2 = self.ffn_norm.forward_gpu(&h, gpu);
            let up = self.ffn_up.forward_gpu(&normed2, gpu);
            let gate = self.ffn_gate.forward_gpu(&normed2, gpu);
            let activated = silu_mul(&gate, &up);
            let ffn_out = self.ffn_down.forward_gpu(&activated, gpu);
            return ctx.add(&h, &ffn_out).unwrap_or_else(|e| {
                log::warn!("GPU add failed, falling back to CPU: {}", e);
                let mut h2 = h;
                h2.add_assign(&ffn_out).unwrap();
                h2
            });
        }
        let _ = gpu;

        let mut h = input.clone();
        h.add_assign(&attn_out).unwrap();
        let normed2 = self.ffn_norm.forward_gpu(&h, gpu);
        let up = self.ffn_up.forward_gpu(&normed2, gpu);
        let gate = self.ffn_gate.forward_gpu(&normed2, gpu);
        let activated = silu_mul(&gate, &up);
        let ffn_out = self.ffn_down.forward_gpu(&activated, gpu);
        h.add_assign(&ffn_out).unwrap();
        h

    }

    /// Helper to create a dummy transformer layer for testing.
    pub fn new_dummy(config: &ModelConfig) -> Self {
        create_dummy_layer(config)
    }
}

pub struct Model {
    pub config: ModelConfig,
    pub embedding: Embedding,
    pub layers: Vec<TransformerLayer>,
    pub norm: RmsNorm,
    pub lm_head: Linear,
    pub cache: Option<KvCache>,
    pub rope_cache: Option<RoPECache>,
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

        let rope_cache = Some(RoPECache::new(
            config.max_seq_len,
            config.head_dim(),
            config.rope_theta,
        ));

        Self {
            config,
            embedding,
            layers,
            norm,
            lm_head,
            cache,
            rope_cache,
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
                hidden = layer.forward_gpu(&hidden, Some(cache), i, pos, gpu, self.rope_cache.as_ref());
            }
            cache.advance(seq_len);
        } else {
            for (i, layer) in self.layers.iter().enumerate() {
                hidden = layer.forward_gpu(&hidden, None, i, pos, gpu, self.rope_cache.as_ref());
            }
        }

        let normed = self.norm.forward_gpu(&hidden, gpu);
        self.lm_head.forward_gpu(&normed, gpu)
    }

    fn logits_to_token(&self, logits: &Tensor, row: usize, sampler: &Sampler) -> u32 {
        let vocab_size = self.config.vocab_size;
        let slice = logits.as_f32_slice();
        let start = row * vocab_size;
        sampler.sample(&slice[start..start + vocab_size])
    }

    pub fn generate(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        sampler: &Sampler,
    ) -> Vec<u32> {
        let mut generated = Vec::new();
        self.generate_loop(prompt_tokens, max_new_tokens, sampler, |t| {
            generated.push(t);
            true
        });
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
        self.generate_loop(prompt_tokens, max_new_tokens, sampler, |t| {
            tx.blocking_send(t).is_ok()
        });
    }

    fn generate_loop(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        sampler: &Sampler,
        mut emit: impl FnMut(u32) -> bool,
    ) {
        let mut tokens = prompt_tokens.to_vec();
        if tokens.is_empty() {
            return;
        }

        #[cfg(feature = "gpu")]
        let gpu_ctx = self.gpu.clone();
        #[cfg(not(feature = "gpu"))]
        let gpu_ctx: Option<GpuContext> = None;

        let logits = self.forward_gpu(&tokens, gpu_ctx.as_ref());
        let last_row = logits.shape()[0] - 1;
        let mut next_token = self.logits_to_token(&logits, last_row, sampler);
        tokens.push(next_token);
        if !emit(next_token) {
            return;
        }

        for _ in 1..max_new_tokens {
            let logits = self.forward_gpu(&[next_token], gpu_ctx.as_ref());
            next_token = self.logits_to_token(&logits, 0, sampler);
            tokens.push(next_token);
            if !emit(next_token) {
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

fn silu_mul(a: &Tensor, b: &Tensor) -> Tensor {
    let mut result = Tensor::zeros(a.shape(), DType::F32);
    let a_slice = a.as_f32_slice();
    let b_slice = b.as_f32_slice();
    let out_slice = result.as_f32_slice_mut();
    bitllm_tensor::simd::f32_silu_mul(a_slice, b_slice, out_slice);
    result
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
        let a = Tensor::from_slice(&[0.0, 1.0, -1.0], &[3]);
        let ones = Tensor::ones(&[3], DType::F32);
        let result = silu_mul(&a, &ones);
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

    #[test]
    fn test_generation_empty_prompt() {
        let config = ModelConfig::tiny_test();
        let mut model = Model::new(config);
        let sampler = Sampler::greedy();
        let generated = model.generate(&[], 5, &sampler);
        assert!(generated.is_empty());
    }

    #[test]
    fn test_generation_deterministic() {
        let config = ModelConfig::tiny_test();
        let mut model1 = Model::new(config.clone());
        let mut model2 = Model::new(config);
        let sampler = Sampler::greedy();
        let gen1 = model1.generate(&[0, 1, 2], 10, &sampler);
        let gen2 = model2.generate(&[0, 1, 2], 10, &sampler);
        assert_eq!(gen1, gen2);
    }

    #[test]
    fn test_forward_single_token() {
        let config = ModelConfig::tiny_test();
        let mut model = Model::new(config);
        let logits = model.forward(&[42]);
        assert_eq!(logits.shape(), &[1, 256]);
    }

    #[test]
    fn test_forward_caches_position() {
        let config = ModelConfig::tiny_test();
        let mut model = Model::new(config);
        model.forward(&[0, 1, 2]);
        let cache_len = model.cache.as_ref().unwrap().get_seq_len();
        assert_eq!(cache_len, 3);
    }

    #[test]
    fn test_clear_cache() {
        let config = ModelConfig::tiny_test();
        let mut model = Model::new(config);
        model.forward(&[0, 1, 2, 3]);
        assert!(model.cache.as_ref().unwrap().get_seq_len() > 0);
        model.clear_cache();
        assert_eq!(model.cache.as_ref().unwrap().get_seq_len(), 0);
    }

    #[test]
    fn test_load_weights_and_generate() {
        let config = ModelConfig::tiny_test();
        let data = create_test_model_safetensors(&config);
        let loader = crate::loader::SafeTensorsLoader::from_bytes(&data).unwrap();

        let mut model = Model::new(config.clone());
        let stats =
            crate::loader::load_safetensors_weights(&mut model, &loader, &config, None);
        assert_eq!(stats.loaded, 20);

        let sampler = Sampler::greedy();
        let generated = model.generate(&[0, 1], 3, &sampler);
        assert_eq!(generated.len(), 3);
    }

    #[test]
    fn test_load_weights_int8_quantize_and_generate() {
        let config = ModelConfig::tiny_test();
        let data = create_test_model_safetensors(&config);
        let loader = crate::loader::SafeTensorsLoader::from_bytes(&data).unwrap();

        let mut model = Model::new(config.clone());
        let stats =
            crate::loader::load_safetensors_weights(&mut model, &loader, &config, Some("int8"));
        assert_eq!(stats.loaded, 20);

        let sampler = Sampler::greedy();
        let generated = model.generate(&[0, 1], 3, &sampler);
        assert_eq!(generated.len(), 3);
    }

    #[test]
    fn test_model_forward_shape_varies_with_seq_len() {
        let config = ModelConfig::tiny_test();
        let mut model = Model::new(config);
        let logits1 = model.forward(&[0]);
        assert_eq!(logits1.shape(), &[1, 256]);
        model.clear_cache();
        let logits3 = model.forward(&[0, 1, 2]);
        assert_eq!(logits3.shape(), &[3, 256]);
    }

    fn create_test_model_safetensors(config: &ModelConfig) -> Vec<u8> {
        let mut tensors: Vec<(String, Vec<f32>, Vec<usize>)> = Vec::new();

        tensors.push((
            "model.embed_tokens.weight".into(),
            vec![0.1; config.vocab_size * config.hidden_size],
            vec![config.vocab_size, config.hidden_size],
        ));

        tensors.push((
            "model.norm.weight".into(),
            vec![1.0; config.hidden_size],
            vec![config.hidden_size],
        ));

        for i in 0..config.num_layers {
            let h = config.hidden_size;
            let kv = config.num_kv_heads() * config.head_dim();
            let inter = config.intermediate_size;

            let layer_tensors = vec![
                (format!("model.layers.{}.self_attn.q_proj.weight", i), vec![0.01; h * kv], vec![kv, h]),
                (format!("model.layers.{}.self_attn.k_proj.weight", i), vec![0.01; h * kv], vec![kv, h]),
                (format!("model.layers.{}.self_attn.v_proj.weight", i), vec![0.01; h * kv], vec![kv, h]),
                (format!("model.layers.{}.self_attn.o_proj.weight", i), vec![0.01; h * kv], vec![h, kv]),
                (format!("model.layers.{}.mlp.gate_proj.weight", i), vec![0.01; inter * h], vec![inter, h]),
                (format!("model.layers.{}.mlp.up_proj.weight", i), vec![0.01; inter * h], vec![inter, h]),
                (format!("model.layers.{}.mlp.down_proj.weight", i), vec![0.01; h * inter], vec![h, inter]),
                (format!("model.layers.{}.input_layernorm.weight", i), vec![1.0; h], vec![h]),
                (format!("model.layers.{}.post_attention_layernorm.weight", i), vec![1.0; h], vec![h]),
            ];
            tensors.extend(layer_tensors);
        }

        let mut header_map = serde_json::Map::new();
        let mut data_blob = Vec::new();
        let mut offset = 0usize;

        for (name, data, shape) in &tensors {
            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
            let len = bytes.len();
            header_map.insert(
                name.clone(),
                serde_json::json!({
                    "dtype": "F32",
                    "shape": shape,
                    "data_offsets": [offset, offset + len]
                }),
            );
            data_blob.extend_from_slice(&bytes);
            offset += len;
        }

        let header = serde_json::Value::Object(header_map);
        let header_str = serde_json::to_string(&header).unwrap();
        let header_bytes = header_str.as_bytes();

        let mut file = Vec::new();
        file.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        file.extend_from_slice(header_bytes);
        file.extend_from_slice(&data_blob);
        file
    }
}
