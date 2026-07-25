use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: Option<usize>,
    pub intermediate_size: usize,
    pub norm_eps: f32,
    pub max_seq_len: usize,
    pub rope_theta: f32,
    pub tie_word_embeddings: bool,
}

impl ModelConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads.unwrap_or(self.num_heads)
    }

    pub fn kv_head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    pub fn num_kv_groups(&self) -> usize {
        self.num_heads / self.num_kv_heads()
    }

    pub fn ff_dim(&self) -> usize {
        self.intermediate_size
    }

    pub fn llama_small() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 4096,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: Some(32),
            intermediate_size: 11008,
            norm_eps: 1e-5,
            max_seq_len: 2048,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
        }
    }

    pub fn tiny_test() -> Self {
        Self {
            vocab_size: 256,
            hidden_size: 64,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: Some(4),
            intermediate_size: 128,
            norm_eps: 1e-5,
            max_seq_len: 128,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
        }
    }
}
