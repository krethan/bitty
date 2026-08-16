use crate::attention::{Attention, KvCache, RoPECache};
use crate::bittransformer::BitTransformerLayer;
use crate::config::{Activation, ModelConfig};
use crate::layers::{Embedding, Linear, RmsNorm};
use crate::sampler::Sampler;
use crate::GpuContext;
use bitllm_quantization::QuantConfig;
use bitllm_tensor::{DType, Tensor};

pub struct TransformerLayer {
    pub attention: Attention,
    pub attn_norm: RmsNorm,
    pub ffn_up: Linear,
    pub ffn_gate: Linear,
    pub ffn_down: Linear,
    pub ffn_norm: RmsNorm,
    /// Gemma-2 post-feedforward RMSNorm, applied to the MLP output before the
    /// residual add.
    pub post_ffn_norm: Option<RmsNorm>,
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
        self.forward_gpu(input, cache, layer_idx, 0, position, None, None)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_gpu(
        &self,
        input: &Tensor,
        cache: Option<&mut KvCache>,
        layer_idx: usize,
        slot: usize,
        position: usize,
        gpu: Option<&GpuContext>,
        rope_cache: Option<&RoPECache>,
    ) -> Tensor {
        let normed = self.attn_norm.forward_gpu(input, gpu);
        let attn_out = self.attention.forward_gpu_with_rope_cache(
            &normed, cache, layer_idx, slot, position, gpu, rope_cache,
        );

        #[cfg(feature = "gpu")]
        if let Some(ctx) = gpu {
            let h = ctx.add(input, &attn_out).unwrap_or_else(|e| {
                log::warn!("GPU add failed, falling back to CPU: {}", e);
                let mut h = input.clone();
                h.add_assign(&attn_out).unwrap();
                h
            });
            let normed2 = self.ffn_norm.forward_gpu(&h, gpu);
            let ffn_out = ffn_forward(
                &self.config,
                &normed2,
                &self.ffn_up,
                &self.ffn_gate,
                &self.ffn_down,
                gpu,
            );
            let ffn_out = self.post_ffn(ffn_out, gpu);
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
        let ffn_out = ffn_forward(
            &self.config,
            &normed2,
            &self.ffn_up,
            &self.ffn_gate,
            &self.ffn_down,
            gpu,
        );
        let ffn_out = self.post_ffn(ffn_out, gpu);
        h.add_assign(&ffn_out).unwrap();
        h
    }

    /// Apply the Gemma-2 post-feedforward norm (if present) to the MLP output.
    fn post_ffn(&self, ffn_out: Tensor, gpu: Option<&GpuContext>) -> Tensor {
        match &self.post_ffn_norm {
            Some(n) => n.forward_gpu(&ffn_out, gpu),
            None => ffn_out,
        }
    }

    /// Batched decode layer forward: `input` is `[batch, hidden_size]` with one
    /// current token per batch slot, RoPE'd and cached at `positions[b]`.
    pub fn forward_batch_gpu(
        &self,
        input: &Tensor,
        cache: Option<&mut KvCache>,
        layer_idx: usize,
        positions: &[usize],
        gpu: Option<&GpuContext>,
        rope_cache: Option<&RoPECache>,
    ) -> Tensor {
        let normed = self.attn_norm.forward_gpu(input, gpu);
        let attn_out = self
            .attention
            .forward_batch_gpu(&normed, cache, layer_idx, positions, gpu, rope_cache);

        let mut h = input.clone();
        h.add_assign(&attn_out).unwrap();
        let normed2 = self.ffn_norm.forward_gpu(&h, gpu);
        let ffn_out = ffn_forward(
            &self.config,
            &normed2,
            &self.ffn_up,
            &self.ffn_gate,
            &self.ffn_down,
            gpu,
        );
        let ffn_out = self.post_ffn(ffn_out, gpu);
        h.add_assign(&ffn_out).unwrap();
        h
    }

    /// FP32 forward that records each quantized projection's `(input, output)`
    /// pair into `recorder` for quantization-aware training. Mirrors the CPU
    /// path of [`TransformerLayer::forward_gpu`].
    pub fn forward_record(
        &self,
        input: &Tensor,
        cache: Option<&mut KvCache>,
        layer_idx: usize,
        position: usize,
        rope_cache: Option<&RoPECache>,
        recorder: &mut crate::record::ProjectionRecorder,
    ) -> Tensor {
        use crate::record::ProjectionKind;

        let normed = self.attn_norm.forward(input);
        let attn_out = self
            .attention
            .forward_record(&normed, cache, layer_idx, position, rope_cache, recorder);

        let mut h = input.clone();
        h.add_assign(&attn_out).unwrap();
        let normed2 = self.ffn_norm.forward(&h);
        let (activated, up, gate) =
            ffn_record_activations(&self.config, &normed2, &self.ffn_up, &self.ffn_gate);
        let mut ffn_out = self.ffn_down.forward(&activated);

        let mut samples = vec![
            (ProjectionKind::Up, &normed2, &up),
            (ProjectionKind::Down, &activated, &ffn_out),
        ];
        if let Some(gate) = &gate {
            samples.insert(1, (ProjectionKind::Gate, &normed2, gate));
        }
        for (kind, input, target) in samples {
            recorder.push(crate::record::ProjectionSample {
                layer: layer_idx,
                kind,
                input: input.clone(),
                target: target.clone(),
            });
        }

        if let Some(post) = &self.post_ffn_norm {
            ffn_out = post.forward(&ffn_out);
        }
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
    /// Learned positional embeddings (GPT-2 `wpe`, Phi `wpe`), indexed by
    /// absolute position. `None` for RoPE-based models.
    pub pos_embedding: Option<Tensor>,
    /// FP32 transformer layers (used when `bit_layers` is None).
    pub layers: Vec<TransformerLayer>,
    /// When set, inference runs through fused 1-bit BitLinear layers.
    pub bit_layers: Option<Vec<BitTransformerLayer>>,
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

        let pos_embedding = config
            .position_embeddings
            .map(|max_pos| Tensor::zeros(&[max_pos, config.hidden_size], DType::F32));

        let norm = make_norm(&config);

        let lm_head = Linear::new(
            Tensor::zeros(&[config.vocab_size, config.hidden_size], DType::F32),
            None,
        );

        let layers = (0..config.num_layers)
            .map(|_| create_dummy_layer(&config))
            .collect();

        let cache = Some(KvCache::new(
            config.num_layers,
            1,
            config.max_seq_len,
            config.num_kv_heads(),
            config.head_dim(),
        ));

        let rope_cache = if config.use_rope {
            let scaling_factor = config.rope_scaling.as_ref().map_or(1.0, |rs| rs.factor);
            Some(RoPECache::with_scaling(
                config.max_seq_len,
                config.head_dim(),
                config.rope_theta,
                scaling_factor,
            ))
        } else {
            None
        };

        Self {
            config,
            embedding,
            pos_embedding,
            layers,
            bit_layers: None,
            norm,
            lm_head,
            cache,
            rope_cache,
            #[cfg(feature = "gpu")]
            gpu: None,
        }
    }

    /// Convert all transformer layers to packed 1-bit BitLinear weights.
    /// Embeddings, norms, and lm_head stay F32. FP32 linear weights are dropped.
    pub fn quantize_to_bit1(&mut self) {
        self.quantize_to_bit1_with_config(&QuantConfig::ternary());
    }

    /// Like [`quantize_to_bit1`], with an explicit quantization config (e.g.
    /// outlier channels via [`QuantConfig::ternary_with_outliers`]).
    pub fn quantize_to_bit1_with_config(&mut self, config: &QuantConfig) {
        self.bit_layers = Some(
            self.layers
                .iter()
                .map(|layer| BitTransformerLayer::from_fp32_layer_q(layer, config))
                .collect(),
        );
        self.layers.clear();
    }

    /// Like [`quantize_to_bit1`], keeping the top `outlier_frac` fraction of
    /// weights per layer exact.
    pub fn quantize_to_bit1_outliers(&mut self, outlier_frac: f64) {
        self.quantize_to_bit1_with_config(&QuantConfig::ternary_with_outliers(outlier_frac));
    }

    /// Like [`quantize_to_bit1_a8`], also enabling per-token int8 activation
    /// quantization on every packed projection.
    pub fn quantize_to_bit1_outliers_a8(&mut self, outlier_frac: f64) {
        self.quantize_to_bit1_outliers(outlier_frac);
        self.enable_a8();
    }

    /// Like [`quantize_to_bit1`], using per-group ternary scales along the
    /// reduction dim. `group_size` must be a multiple of 8.
    pub fn quantize_to_bit1_grouped(&mut self, group_size: usize) {
        self.quantize_to_bit1_with_config(&QuantConfig::ternary_grouped(group_size));
    }

    /// Like [`quantize_to_bit1_grouped`], also enabling per-token int8
    /// activation quantization on every packed projection.
    pub fn quantize_to_bit1_grouped_a8(&mut self, group_size: usize) {
        self.quantize_to_bit1_grouped(group_size);
        self.enable_a8();
    }

    /// Like [`quantize_to_bit1`], but also enables per-token int8 activation
    /// quantization (W1A8) on every packed projection.
    ///
    /// W1A8 is now the default quantized path, so this is a no-op kept for
    /// backwards compatibility and explicitness.
    pub fn quantize_to_bit1_a8(&mut self) {
        self.quantize_to_bit1();
        self.enable_a8();
    }

    fn enable_a8(&mut self) {
        if let Some(ref mut bit_layers) = self.bit_layers {
            for layer in bit_layers.iter_mut() {
                layer.attention.q_proj.a8 = true;
                layer.attention.k_proj.a8 = true;
                layer.attention.v_proj.a8 = true;
                layer.attention.o_proj.a8 = true;
                layer.ffn_up.a8 = true;
                layer.ffn_gate.a8 = true;
                layer.ffn_down.a8 = true;
            }
        }
    }

    pub fn is_bit1(&self) -> bool {
        self.bit_layers.is_some()
    }

    /// Tie word embeddings: copy the embedding weight to lm_head.
    /// Used by models like LLaMA-2 (7B), Gemma, and T5 where the embedding
    /// and output projection share weights.
    pub fn tie_embeddings(&mut self) {
        self.lm_head.weight = self.embedding.weight.clone();
    }

    /// Add learned position embeddings (GPT-2/Phi `wpe`) in place, for a
    /// contiguous sequence of rows starting at absolute `pos`. No-op for
    /// RoPE-based models (`pos_embedding` is `None`).
    fn add_position_embedding_inplace(&self, hidden: &mut Tensor, pos: usize) {
        if let Some(ref pe) = self.pos_embedding {
            let hidden_size = self.config.hidden_size;
            let max_pos = self.config.position_embeddings.unwrap_or(0);
            let rows = hidden.shape()[0];
            let h = hidden.as_f32_slice_mut();
            let p = pe.as_f32_slice();
            for r in 0..rows {
                let src = (pos + r).min(max_pos.saturating_sub(1));
                let base = r * hidden_size;
                let pbase = src * hidden_size;
                for j in 0..hidden_size {
                    h[base + j] += p[pbase + j];
                }
            }
        }
    }

    /// Add learned position embeddings in place for a batch of tokens, where
    /// `positions[b]` is the absolute position of token `b`. No-op for
    /// RoPE-based models.
    fn add_position_embedding_batch_inplace(&self, hidden: &mut Tensor, positions: &[usize]) {
        if let Some(ref pe) = self.pos_embedding {
            let hidden_size = self.config.hidden_size;
            let max_pos = self.config.position_embeddings.unwrap_or(0);
            let h = hidden.as_f32_slice_mut();
            let p = pe.as_f32_slice();
            for (r, &pos) in positions.iter().enumerate() {
                let src = pos.min(max_pos.saturating_sub(1));
                let base = r * hidden_size;
                let pbase = src * hidden_size;
                for j in 0..hidden_size {
                    h[base + j] += p[pbase + j];
                }
            }
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
        self.forward_slot(token_ids, 0, gpu)
    }

    /// Forward a sequence of tokens into a specific cache slot. The model's
    /// cache must have been sized for at least `slot + 1` batch entries.
    pub fn forward_slot(
        &mut self,
        token_ids: &[u32],
        slot: usize,
        gpu: Option<&GpuContext>,
    ) -> Tensor {
        let normed = self.forward_normed(token_ids, slot, gpu);
        let mut logits = self.lm_head.forward_gpu(&normed, gpu);
        self.apply_final_softcap(&mut logits);
        logits
    }

    /// Gemma-2 final logit soft-capping: `cap * tanh(logit / cap)`.
    fn apply_final_softcap(&self, logits: &mut Tensor) {
        let cap = self.config.final_logit_softcap();
        if cap <= 0.0 {
            return;
        }
        for s in logits.as_f32_slice_mut().iter_mut() {
            *s = cap * (*s / cap).tanh();
        }
    }

    /// Like `forward_slot`, but returns the post-RMSNorm hidden states (the
    /// `lm_head` input) instead of the logits. Exposes the readout input so
    /// probes/trainers can evaluate the head on the actual hidden states.
    pub fn forward_hidden(
        &mut self,
        token_ids: &[u32],
        slot: usize,
        gpu: Option<&GpuContext>,
    ) -> Tensor {
        self.forward_normed(token_ids, slot, gpu)
    }

    /// Like `forward_hidden`, but records each quantized projection's
    /// `(input, fp32 output)` pair into `recorder` (QAT teacher pass). Runs the
    /// FP32 transformer layers; only meaningful before quantization.
    pub fn forward_record(
        &mut self,
        token_ids: &[u32],
        recorder: &mut crate::record::ProjectionRecorder,
    ) -> Tensor {
        let seq_len = token_ids.len();
        let pos = self.cache.as_ref().map_or(0, |c| c.seq_len(0));
        if let Some(ref mut cache) = self.cache {
            cache.reserve(0, pos, seq_len);
        }

        let mut hidden = self.embedding.forward(token_ids);
        self.add_position_embedding_inplace(&mut hidden, pos);
        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward_record(
                &hidden,
                self.cache.as_mut(),
                i,
                pos,
                self.rope_cache.as_ref(),
                recorder,
            );
        }
        self.norm.forward(&hidden)
    }

    fn forward_normed(
        &mut self,
        token_ids: &[u32],
        slot: usize,
        gpu: Option<&GpuContext>,
    ) -> Tensor {
        let seq_len = token_ids.len();
        let pos = self.cache.as_ref().map_or(0, |c| c.seq_len(slot));

        // Mark the positions being written as populated before the layer loop
        // so attention sees the full range.
        if let Some(ref mut cache) = self.cache {
            cache.reserve(slot, pos, seq_len);
        }

        let mut hidden = self.embedding.forward(token_ids);
        self.add_position_embedding_inplace(&mut hidden, pos);

        if let Some(ref mut cache) = self.cache {
            if let Some(ref bit_layers) = self.bit_layers {
                for (i, layer) in bit_layers.iter().enumerate() {
                    hidden = layer.forward_gpu(
                        &hidden,
                        Some(cache),
                        i,
                        slot,
                        pos,
                        gpu,
                        self.rope_cache.as_ref(),
                    );
                }
            } else {
                for (i, layer) in self.layers.iter().enumerate() {
                    hidden = layer.forward_gpu(
                        &hidden,
                        Some(cache),
                        i,
                        slot,
                        pos,
                        gpu,
                        self.rope_cache.as_ref(),
                    );
                }
            }
        } else if let Some(ref bit_layers) = self.bit_layers {
            for (i, layer) in bit_layers.iter().enumerate() {
                hidden =
                    layer.forward_gpu(&hidden, None, i, slot, pos, gpu, self.rope_cache.as_ref());
            }
        } else {
            for (i, layer) in self.layers.iter().enumerate() {
                hidden =
                    layer.forward_gpu(&hidden, None, i, slot, pos, gpu, self.rope_cache.as_ref());
            }
        }

        self.norm.forward_gpu(&hidden, gpu)
    }

    /// Batched decode step. `next_tokens[b]` is the current token for batch
    /// slot `b`, to be written into the cache at absolute `positions[b]`.
    /// Returns `[batch, vocab_size]` logits, one row per slot.
    pub fn forward_batch_decode(
        &mut self,
        next_tokens: &[u32],
        positions: &[usize],
        gpu: Option<&GpuContext>,
    ) -> Tensor {
        let mut hidden = self.embedding.forward(next_tokens);
        self.add_position_embedding_batch_inplace(&mut hidden, positions);

        if let Some(ref mut cache) = self.cache {
            for (b, &pos) in positions.iter().enumerate() {
                cache.reserve(b, pos, 1);
            }

            if let Some(ref bit_layers) = self.bit_layers {
                for (i, layer) in bit_layers.iter().enumerate() {
                    hidden = layer.forward_batch_gpu(&hidden, Some(cache), i, positions, gpu);
                }
            } else {
                for (i, layer) in self.layers.iter().enumerate() {
                    hidden = layer.forward_batch_gpu(
                        &hidden,
                        Some(cache),
                        i,
                        positions,
                        gpu,
                        self.rope_cache.as_ref(),
                    );
                }
            }

            let normed = self.norm.forward_gpu(&hidden, gpu);
            let mut logits = self.lm_head.forward_gpu(&normed, gpu);
            self.apply_final_softcap(&mut logits);
            return logits;
        }

        if let Some(ref bit_layers) = self.bit_layers {
            for (i, layer) in bit_layers.iter().enumerate() {
                hidden = layer.forward_batch_gpu(&hidden, None, i, positions, gpu);
            }
        } else {
            for (i, layer) in self.layers.iter().enumerate() {
                hidden = layer.forward_batch_gpu(
                    &hidden,
                    None,
                    i,
                    positions,
                    gpu,
                    self.rope_cache.as_ref(),
                );
            }
        }

        let normed = self.norm.forward_gpu(&hidden, gpu);
        let mut logits = self.lm_head.forward_gpu(&normed, gpu);
        self.apply_final_softcap(&mut logits);
        logits
    }

    /// Prefill a batch of prompts into their respective cache slots.
    /// Each prompt is processed in parallel, populating the KV cache.
    /// Returns the logits for the last token of each prompt (for sampling).
    ///
    /// This is the compute-bound phase: each prompt can be long, and we
    /// process all tokens in parallel. Benefits from large batch sizes.
    pub fn prefill_batch(&mut self, prompts: &[&[u32]], gpu: Option<&GpuContext>) -> Tensor {
        let batch = prompts.len();
        self.ensure_cache_batch(batch);

        // Process each prompt into its cache slot
        let mut last_logits = Vec::with_capacity(batch);
        for (b, prompt) in prompts.iter().enumerate() {
            if prompt.is_empty() {
                // Empty prompt: just get embedding for token 0
                let dummy = [0u32];
                let logits = self.forward_slot(&dummy, b, gpu);
                last_logits.push(logits);
            } else {
                let logits = self.forward_slot(prompt, b, gpu);
                last_logits.push(logits);
            }
        }

        // Stack logits from all slots into a single [batch, vocab] tensor
        let vocab = self.config.vocab_size;
        let mut stacked = Tensor::zeros(&[batch, vocab], DType::F32);
        let out = stacked.as_f32_slice_mut();
        for (b, logits) in last_logits.iter().enumerate() {
            let src = logits.as_f32_slice();
            let last_row = logits.shape()[0] - 1;
            let src_start = last_row * vocab;
            let dst_start = b * vocab;
            out[dst_start..dst_start + vocab].copy_from_slice(&src[src_start..src_start + vocab]);
        }
        stacked
    }

    /// Decode step for a batch of sequences. Each sequence contributes one
    /// token. Returns `[batch, vocab_size]` logits.
    ///
    /// This is the memory-bound phase: we read the entire KV cache for each
    /// token. Benefits from batching multiple sequences to amortize cache reads.
    pub fn decode_batch(&mut self, tokens: &[u32], gpu: Option<&GpuContext>) -> Tensor {
        let batch = tokens.len();
        let positions: Vec<usize> = (0..batch)
            .map(|b| self.cache.as_ref().map_or(0, |c| c.seq_len(b)))
            .collect();

        self.forward_batch_decode(tokens, &positions, gpu)
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
            cache.clear();
        }
    }

    /// (Re)size the KV cache to hold `batch` sequences. Reuses the existing
    /// cache when possible, otherwise rebuilds it with the requested batch.
    fn ensure_cache_batch(&mut self, batch: usize) {
        let reusable = self.cache.as_ref().is_some_and(|c| c.batch == batch);
        if reusable {
            self.clear_cache();
            return;
        }
        self.cache = Some(KvCache::new(
            self.config.num_layers,
            batch,
            self.config.max_seq_len,
            self.config.num_kv_heads(),
            self.config.head_dim(),
        ));
    }

    /// Public wrapper around [`Model::ensure_cache_batch`]: size the KV cache
    /// for `capacity` concurrent sequences (used by continuous batching).
    pub fn reserve_cache_batch(&mut self, capacity: usize) {
        self.ensure_cache_batch(capacity.max(1));
    }

    /// Roll slot `slot`'s populated KV length back to `len` (speculative
    /// decoding discards rejected draft tokens this way).
    pub fn truncate_cache(&mut self, slot: usize, len: usize) {
        if let Some(cache) = self.cache.as_mut() {
            cache.truncate(slot, len);
        }
    }

    /// Greedy/temperature generation for a batch of independent sequences.
    /// Each prompt is prefilled into its own cache slot, then all sequences
    /// decode in lockstep. Returns the generated tokens for each sequence
    /// (prompts are not included). A sequence stops early when it produces
    /// `eos`, if provided.
    ///
    /// Uses the prefill/decode separation: `prefill_batch` for the prompt
    /// phase, `decode_batch` for the generation phase.
    pub fn generate_batch(
        &mut self,
        prompts: &[&[u32]],
        max_new_tokens: usize,
        sampler: &Sampler,
        eos: Option<u32>,
    ) -> Vec<Vec<u32>> {
        let n = prompts.len();
        let mut outputs: Vec<Vec<u32>> = (0..n).map(|_| Vec::new()).collect();
        if n == 0 || max_new_tokens == 0 {
            return outputs;
        }

        #[cfg(feature = "gpu")]
        let gpu_ctx = self.gpu.clone();
        #[cfg(not(feature = "gpu"))]
        let gpu_ctx: Option<GpuContext> = None;

        // Prefill phase: process all prompts in parallel
        let mut logits = self.prefill_batch(prompts, gpu_ctx.as_ref());
        let vocab = self.config.vocab_size;

        let mut next: Vec<u32> = vec![0; n];
        let mut done = vec![false; n];
        let mut finished = 0usize;

        // Decode phase: generate tokens one at a time for all sequences
        for _ in 0..max_new_tokens {
            // Sample from current logits
            let slice = logits.as_f32_slice();
            for b in 0..n {
                if done[b] {
                    continue;
                }
                let start = b * vocab;
                let token = sampler.sample(&slice[start..start + vocab]);
                next[b] = token;
                outputs[b].push(token);
                if Some(token) == eos {
                    done[b] = true;
                    finished += 1;
                }
            }

            if finished == n {
                break;
            }

            // Process sampled tokens through model to get next logits
            logits = self.decode_batch(&next, gpu_ctx.as_ref());
        }

        outputs
    }

    pub fn generate_streaming(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        sampler: &Sampler,
        tx: tokio::sync::mpsc::Sender<u32>,
        token_count: &mut u64,
    ) {
        self.generate_loop(prompt_tokens, max_new_tokens, sampler, |t| {
            *token_count += 1;
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

    let qk_norm = if config.qk_norm {
        Some(make_norm_shape(&[config.num_heads, head_dim], config))
    } else {
        None
    };

    TransformerLayer {
        attention: Attention::new_with_qk_norm(
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
            qk_norm.clone(),
            qk_norm,
            config.clone(),
        ),
        attn_norm: make_norm(config),
        ffn_up: Linear::new(Tensor::zeros(&[config.ff_dim(), hidden], DType::F32), None),
        ffn_gate: Linear::new(Tensor::zeros(&[config.ff_dim(), hidden], DType::F32), None),
        ffn_down: Linear::new(Tensor::zeros(&[hidden, config.ff_dim()], DType::F32), None),
        ffn_norm: make_norm(config),
        post_ffn_norm: if config.post_ffn_norm {
            Some(make_norm(config))
        } else {
            None
        },
        config: config.clone(),
    }
}

/// Build the pre-attention / pre-FFN norm for a config: RMSNorm or LayerNorm.
fn make_norm(config: &ModelConfig) -> RmsNorm {
    make_norm_shape(&[config.hidden_size], config)
}

fn make_norm_shape(shape: &[usize], config: &ModelConfig) -> RmsNorm {
    if config.uses_layer_norm() {
        RmsNorm::new_layer(
            Tensor::ones(shape, DType::F32),
            Some(Tensor::zeros(shape, DType::F32)),
            config.norm_eps,
        )
    } else if config.one_centered_norm {
        RmsNorm::new_one_centered(Tensor::ones(shape, DType::F32), config.norm_eps)
    } else {
        RmsNorm::new(Tensor::ones(shape, DType::F32), config.norm_eps)
    }
}

fn ffn_forward(
    config: &ModelConfig,
    normed2: &Tensor,
    up: &Linear,
    gate: &Linear,
    down: &Linear,
    gpu: Option<&GpuContext>,
) -> Tensor {
    match config.default_activation() {
        Activation::SiluGated => {
            let up_out = up.forward_gpu(normed2, gpu);
            let gate_out = gate.forward_gpu(normed2, gpu);
            down.forward_gpu(&silu_mul(&gate_out, &up_out), gpu)
        }
        Activation::GeluGated => {
            let up_out = up.forward_gpu(normed2, gpu);
            let gate_out = gate.forward_gpu(normed2, gpu);
            let mut gate_gelu = Tensor::zeros(gate_out.shape(), DType::F32);
            bitllm_tensor::simd::f32_gelu_tanh(
                gate_out.as_f32_slice(),
                gate_gelu.as_f32_slice_mut(),
            );
            let mut activated = Tensor::zeros(gate_out.shape(), DType::F32);
            bitllm_tensor::simd::f32_mul(
                gate_gelu.as_f32_slice(),
                up_out.as_f32_slice(),
                activated.as_f32_slice_mut(),
            );
            down.forward_gpu(&activated, gpu)
        }
        Activation::Gelu => {
            let up_out = up.forward_gpu(normed2, gpu);
            let mut activated = Tensor::zeros(up_out.shape(), DType::F32);
            bitllm_tensor::simd::f32_gelu(up_out.as_f32_slice(), activated.as_f32_slice_mut());
            down.forward_gpu(&activated, gpu)
        }
    }
}

/// CPU record-path FFN: returns `(activated, up_out, gate_out)`. `gate_out` is
/// `None` for non-gated GELU models (GPT-2).
fn ffn_record_activations(
    config: &ModelConfig,
    normed2: &Tensor,
    up: &Linear,
    gate: &Linear,
) -> (Tensor, Tensor, Option<Tensor>) {
    match config.default_activation() {
        Activation::SiluGated => {
            let up_out = up.forward(normed2);
            let gate_out = gate.forward(normed2);
            let activated = silu_mul(&gate_out, &up_out);
            (activated, up_out, Some(gate_out))
        }
        Activation::GeluGated => {
            let up_out = up.forward(normed2);
            let gate_out = gate.forward(normed2);
            let mut gate_gelu = Tensor::zeros(gate_out.shape(), DType::F32);
            bitllm_tensor::simd::f32_gelu_tanh(
                gate_out.as_f32_slice(),
                gate_gelu.as_f32_slice_mut(),
            );
            let mut activated = Tensor::zeros(gate_out.shape(), DType::F32);
            bitllm_tensor::simd::f32_mul(
                gate_gelu.as_f32_slice(),
                up_out.as_f32_slice(),
                activated.as_f32_slice_mut(),
            );
            (activated, up_out, Some(gate_out))
        }
        Activation::Gelu => {
            let up_out = up.forward(normed2);
            let mut activated = Tensor::zeros(up_out.shape(), DType::F32);
            bitllm_tensor::simd::f32_gelu(up_out.as_f32_slice(), activated.as_f32_slice_mut());
            (activated, up_out, None)
        }
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
    fn test_generate_batch_matches_single_sequence() {
        // A batch of one must reproduce the single-sequence path exactly.
        let config = ModelConfig::tiny_test();
        let sampler = Sampler::greedy();

        let mut single = Model::new(config.clone());
        let expected = single.generate(&[0, 1, 2, 5], 6, &sampler);

        let mut batched = Model::new(config);
        let prompts = [vec![0u32, 1, 2, 5]];
        let refs: Vec<&[u32]> = prompts.iter().map(|v| v.as_slice()).collect();
        let got = batched.generate_batch(&refs, 6, &sampler, None);

        assert_eq!(got.len(), 1);
        assert_eq!(got[0], expected);
    }

    #[test]
    fn test_generate_batch_multiple_sequences() {
        let config = ModelConfig::tiny_test();
        let mut model = Model::new(config);
        let sampler = Sampler::greedy();

        let p0 = [0u32, 1];
        let p1 = [2u32, 3, 4];
        let prompts = [p0.as_slice(), p1.as_slice()];
        let outputs = model.generate_batch(&prompts, 5, &sampler, None);

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].len(), 5);
        assert_eq!(outputs[1].len(), 5);

        // Each slot must have independently advanced its cache.
        let cache = model.cache.as_ref().unwrap();
        assert_eq!(cache.seq_len(0), 2 + 5);
        assert_eq!(cache.seq_len(1), 3 + 5);
    }

    #[test]
    fn test_generate_batch_eos_stops_sequence() {
        // The zero-initialized tiny model has all-equal logits, so greedy
        // always samples token 0. With eos = Some(0) each sequence must stop
        // after its very first generated token.
        let config = ModelConfig::tiny_test();
        let mut model = Model::new(config.clone());
        let sampler = Sampler::greedy();

        let p0 = [0u32, 1];
        let p1 = [2u32, 3];
        let prompts = [p0.as_slice(), p1.as_slice()];
        let outputs = model.generate_batch(&prompts, 8, &sampler, Some(0));

        assert_eq!(outputs.len(), 2);
        for seq in &outputs {
            assert_eq!(seq.len(), 1, "EOS should stop after the first token");
            assert_eq!(seq[0], 0);
        }

        // A different EOS never fires, so sequences run to max_new_tokens.
        let mut model = Model::new(config.clone());
        let outputs = model.generate_batch(&prompts, 8, &sampler, Some(2));
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].len(), 8);
        assert_eq!(outputs[1].len(), 8);
    }

    #[test]
    fn test_generate_batch_bit1() {
        let config = ModelConfig::tiny_test();
        let data = create_test_model_safetensors(&config);
        let loader = crate::loader::SafeTensorsLoader::from_bytes(&data).unwrap();
        let mut model = Model::new(config.clone());
        let stats =
            crate::loader::load_safetensors_weights(&mut model, &loader, &config, Some("ternary"));
        assert_eq!(stats.loaded, 20);
        assert!(model.is_bit1());

        let p0 = [0u32, 1];
        let p1 = [2u32, 3];
        let prompts = [p0.as_slice(), p1.as_slice()];
        let sampler = Sampler::greedy();
        let outputs = model.generate_batch(&prompts, 4, &sampler, None);
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].len(), 4);
        assert_eq!(outputs[1].len(), 4);
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
        let cache_len = model.cache.as_ref().unwrap().seq_len(0);
        assert_eq!(cache_len, 3);
    }

    #[test]
    fn test_clear_cache() {
        let config = ModelConfig::tiny_test();
        let mut model = Model::new(config);
        model.forward(&[0, 1, 2, 3]);
        assert!(model.cache.as_ref().unwrap().seq_len(0) > 0);
        model.clear_cache();
        assert_eq!(model.cache.as_ref().unwrap().seq_len(0), 0);
    }

    #[test]
    fn test_load_weights_and_generate() {
        let config = ModelConfig::tiny_test();
        let data = create_test_model_safetensors(&config);
        let loader = crate::loader::SafeTensorsLoader::from_bytes(&data).unwrap();

        let mut model = Model::new(config.clone());
        let stats = crate::loader::load_safetensors_weights(&mut model, &loader, &config, None);
        assert_eq!(stats.loaded, 20);
        assert!(!model.is_bit1());

        let sampler = Sampler::greedy();
        let generated = model.generate(&[0, 1], 3, &sampler);
        assert_eq!(generated.len(), 3);
    }

    #[test]
    fn test_load_weights_ternary_bit1_and_generate() {
        let config = ModelConfig::tiny_test();
        let data = create_test_model_safetensors(&config);
        let loader = crate::loader::SafeTensorsLoader::from_bytes(&data).unwrap();

        let mut model = Model::new(config.clone());
        let stats =
            crate::loader::load_safetensors_weights(&mut model, &loader, &config, Some("ternary"));
        assert_eq!(stats.loaded, 20);
        assert!(model.is_bit1());

        let logits = model.forward(&[0, 1, 2]);
        assert_eq!(logits.shape(), &[3, 256]);

        model.clear_cache();
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
                (
                    format!("model.layers.{}.self_attn.q_proj.weight", i),
                    vec![0.01; h * kv],
                    vec![kv, h],
                ),
                (
                    format!("model.layers.{}.self_attn.k_proj.weight", i),
                    vec![0.01; h * kv],
                    vec![kv, h],
                ),
                (
                    format!("model.layers.{}.self_attn.v_proj.weight", i),
                    vec![0.01; h * kv],
                    vec![kv, h],
                ),
                (
                    format!("model.layers.{}.self_attn.o_proj.weight", i),
                    vec![0.01; h * kv],
                    vec![h, kv],
                ),
                (
                    format!("model.layers.{}.mlp.gate_proj.weight", i),
                    vec![0.01; inter * h],
                    vec![inter, h],
                ),
                (
                    format!("model.layers.{}.mlp.up_proj.weight", i),
                    vec![0.01; inter * h],
                    vec![inter, h],
                ),
                (
                    format!("model.layers.{}.mlp.down_proj.weight", i),
                    vec![0.01; h * inter],
                    vec![h, inter],
                ),
                (
                    format!("model.layers.{}.input_layernorm.weight", i),
                    vec![1.0; h],
                    vec![h],
                ),
                (
                    format!("model.layers.{}.post_attention_layernorm.weight", i),
                    vec![1.0; h],
                    vec![h],
                ),
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

    #[test]
    fn test_tie_word_embeddings() {
        let mut config = ModelConfig::tiny_test();
        config.tie_word_embeddings = true;
        let mut model = Model::new(config);

        // Set embedding to a known pattern
        let emb_data: Vec<f32> = (0..model.embedding.weight.num_elements())
            .map(|i| i as f32 * 0.001)
            .collect();
        model.embedding.weight = Tensor::from_slice(&emb_data, model.embedding.weight.shape());

        // Tie embeddings
        model.tie_embeddings();

        // Verify lm_head weight matches embedding weight
        assert_eq!(model.lm_head.weight.shape(), model.embedding.weight.shape());
        let lm_data = model.lm_head.weight.as_f32_slice();
        let emb_data = model.embedding.weight.as_f32_slice();
        assert_eq!(lm_data, emb_data);

        // Verify forward pass uses tied weights (logits should be based on embedding)
        let logits = model.forward(&[0, 1]);
        assert_eq!(logits.shape(), &[2, 256]); // vocab_size = 256 for tiny_test
    }
}
