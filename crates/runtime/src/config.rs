use serde::{Deserialize, Serialize};
use serde_json::Value;

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
    /// BitNet b1.58-style mixed-precision residual: when true, quantized
    /// blocks use `SubLN(x) = x - RMSNorm(x)` as their input (bounded
    /// activations for W1A8) and add the residual back in f32. The residual
    /// stream is never quantized.
    #[serde(default)]
    pub sub_ln: bool,
    /// RoPE scaling configuration for extended context (LLaMA-2/3).
    /// `None` means no scaling (standard RoPE).
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
}

/// RoPE scaling configuration for extended context windows.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeScaling {
    /// Scaling type: "linear", "dynamic", or "llama3" (LLaMA-3 specific).
    pub r#type: String,
    /// Scaling factor (e.g., 2.0 for 2x context extension).
    pub factor: f32,
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
            sub_ln: false,
            rope_scaling: None,
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
            sub_ln: false,
            rope_scaling: None,
        }
    }

    /// Parse a HuggingFace `config.json` into a `ModelConfig`.
    ///
    /// HuggingFace uses different field names than our internal format:
    /// - `num_hidden_layers` → `num_layers`
    /// - `num_attention_heads` → `num_heads`
    /// - `num_key_value_heads` → `num_kv_heads`
    /// - `intermediate_size` → `intermediate_size` (same)
    /// - `rms_norm_eps` → `norm_eps`
    /// - `max_position_embeddings` → `max_seq_len`
    pub fn from_huggingface_json(json: &str) -> Result<Self, String> {
        let v: Value = serde_json::from_str(json).map_err(|e| format!("JSON parse error: {}", e))?;

        let get_u64 = |key: &str| -> Option<usize> {
            v.get(key).and_then(|v| v.as_u64()).map(|n| n as usize)
        };
        let get_f32 = |key: &str| -> Option<f32> {
            v.get(key).and_then(|v| v.as_f64()).map(|n| n as f32)
        };
        let get_bool = |key: &str| -> Option<bool> {
            v.get(key).and_then(|v| v.as_bool())
        };

        let vocab_size = get_u64("vocab_size").unwrap_or(32000);
        let hidden_size = get_u64("hidden_size").ok_or("missing `hidden_size`")?;
        let num_layers = get_u64("num_hidden_layers")
            .or_else(|| get_u64("num_layers"))
            .ok_or("missing `num_hidden_layers`")?;
        let num_heads = get_u64("num_attention_heads")
            .or_else(|| get_u64("num_heads"))
            .ok_or("missing `num_attention_heads`")?;
        let num_kv_heads = get_u64("num_key_value_heads").map(Some).unwrap_or(None);
        let intermediate_size = get_u64("intermediate_size")
            .unwrap_or_else(|| hidden_size * 4);
        let norm_eps = get_f32("rms_norm_eps")
            .or_else(|| get_f32("layer_norm_eps"))
            .unwrap_or(1e-5);
        let max_seq_len = get_u64("max_position_embeddings")
            .or_else(|| get_u64("max_seq_len"))
            .unwrap_or(2048);
        let rope_theta = get_f32("rope_theta").unwrap_or(10000.0);
        let tie_word_embeddings = get_bool("tie_word_embeddings").unwrap_or(false);
        let sub_ln = get_bool("sub_ln").unwrap_or(false);

        // Parse rope_scaling as an object with "type" and "factor" fields
        let rope_scaling = v.get("rope_scaling").and_then(|rs| {
            if rs.is_object() {
                let rtype = rs.get("type").and_then(|t| t.as_str()).unwrap_or("linear").to_string();
                let factor = rs.get("factor").and_then(|f| f.as_f64()).unwrap_or(1.0) as f32;
                Some(RopeScaling { r#type: rtype, factor })
            } else {
                None
            }
        });

        Ok(Self {
            vocab_size,
            hidden_size,
            num_layers,
            num_heads,
            num_kv_heads,
            intermediate_size,
            norm_eps,
            max_seq_len,
            rope_theta,
            tie_word_embeddings,
            sub_ln,
            rope_scaling,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_scaling_parsing() {
        let json = r#"{
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "intermediate_size": 11008,
            "rms_norm_eps": 1e-5,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "rope_scaling": {
                "type": "linear",
                "factor": 2.0
            }
        }"#;

        let config = ModelConfig::from_huggingface_json(json).unwrap();
        assert!(config.rope_scaling.is_some());
        let rs = config.rope_scaling.unwrap();
        assert_eq!(rs.r#type, "linear");
        assert_eq!(rs.factor, 2.0);
    }

    #[test]
    fn test_rope_scaling_missing() {
        let json = r#"{
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "intermediate_size": 11008,
            "rms_norm_eps": 1e-5,
            "max_position_embeddings": 2048,
            "rope_theta": 10000.0
        }"#;

        let config = ModelConfig::from_huggingface_json(json).unwrap();
        assert!(config.rope_scaling.is_none());
    }
}
