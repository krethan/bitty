use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Architecture {
    #[default]
    Llama,
    Mistral,
    Gpt2,
    Phi,
    Gemma,
    Qwen2,
    Qwen3,
    Custom(String),
}

impl std::fmt::Display for Architecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Architecture::Llama => write!(f, "Llama"),
            Architecture::Mistral => write!(f, "Mistral"),
            Architecture::Gpt2 => write!(f, "Gpt2"),
            Architecture::Phi => write!(f, "Phi"),
            Architecture::Gemma => write!(f, "Gemma"),
            Architecture::Qwen2 => write!(f, "Qwen2"),
            Architecture::Qwen3 => write!(f, "Qwen3"),
            Architecture::Custom(s) => write!(f, "{}", s),
        }
    }
}

impl Architecture {
    pub fn from_huggingface(config: &Value) -> Option<Self> {
        if let Some(archs) = config.get("architectures").and_then(|v| v.as_array()) {
            for arch in archs {
                if let Some(s) = arch.as_str() {
                    return Some(match s {
                        "LlamaForCausalLM" | "LlamaModel" => Architecture::Llama,
                        "MistralForCausalLM" | "MistralModel" => Architecture::Mistral,
                        "GPT2LMHeadModel" | "GPT2Model" => Architecture::Gpt2,
                        "PhiForCausalLM" | "PhiModel" => Architecture::Phi,
                        "GemmaForCausalLM" | "GemmaModel" | "Gemma2ForCausalLM"
                        | "Gemma2Model" => Architecture::Gemma,
                        "Qwen2ForCausalLM" | "Qwen2Model" => Architecture::Qwen2,
                        "Qwen3ForCausalLM" | "Qwen3Model" => Architecture::Qwen3,
                        other => Architecture::Custom(other.to_string()),
                    });
                }
            }
        }
        config.get("model_type").and_then(|v| v.as_str()).map(|s| match s {
            "llama" => Architecture::Llama,
            "mistral" => Architecture::Mistral,
            "gpt2" => Architecture::Gpt2,
            "phi" => Architecture::Phi,
            "gemma" | "gemma2" => Architecture::Gemma,
            "qwen2" => Architecture::Qwen2,
            "qwen3" => Architecture::Qwen3,
            other => Architecture::Custom(other.to_string()),
        })
    }

    /// Map a GGUF `general.architecture` string to an `Architecture`.
    pub fn from_gguf(s: &str) -> Self {
        match s {
            "llama" => Architecture::Llama,
            "mistral" => Architecture::Mistral,
            "gpt2" => Architecture::Gpt2,
            "phi" | "phi2" => Architecture::Phi,
            "gemma" | "gemma2" => Architecture::Gemma,
            "qwen2" => Architecture::Qwen2,
            "qwen3" => Architecture::Qwen3,
            other => Architecture::Custom(other.to_string()),
        }
    }

    /// Whether this architecture uses RoPE rather than learned positional
    /// embeddings. GPT-2 and Phi embed position information directly.
    pub fn uses_rope(&self) -> bool {
        !matches!(self, Architecture::Gpt2 | Architecture::Phi)
    }

    /// Whether this architecture uses RMSNorm (`false` → LayerNorm).
    /// GPT-2 and Phi use LayerNorm with a bias term.
    pub fn uses_rms_norm(&self) -> bool {
        !matches!(self, Architecture::Gpt2 | Architecture::Phi)
    }

    /// The FFN shape/activation convention for this architecture.
    pub fn default_activation(&self) -> Activation {
        match self {
            Architecture::Gpt2 | Architecture::Phi => Activation::Gelu,
            Architecture::Gemma => Activation::GeluGated,
            _ => Activation::SiluGated,
        }
    }
}

/// FFN activation convention.
///
/// - `SiluGated`: SwiGLU `down(silu(gate) * up)` (LLaMA/Mistral/Qwen)
/// - `GeluGated`: `down(gelu(gate) * up)` (Gemma)
/// - `Gelu`: single-FC `down(gelu(up))` (GPT-2, Phi-1/2)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Activation {
    #[default]
    SiluGated,
    GeluGated,
    Gelu,
}

impl Activation {
    pub fn is_gated(&self) -> bool {
        !matches!(self, Activation::Gelu)
    }
}

/// Normalization layer convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum NormType {
    #[default]
    RmsNorm,
    LayerNorm,
}

/// Normalize `hidden_act`/`hidden_activation`/`activation_function` strings
/// from a HF config into an [`Activation`].
fn parse_activation(s: &str, arch: &Architecture) -> Activation {
    match s {
        "silu" | "swish" => Activation::SiluGated,
        "gelu" => match arch {
            // Gemma config.json uses `hidden_act: "gelu_pytorch_tanh"`; a bare
            // "gelu" on a Gemma checkpoint means the gated variant.
            Architecture::Gemma => Activation::GeluGated,
            // GPT-2/Phi: single-FC GELU.
            Architecture::Gpt2 | Architecture::Phi => Activation::Gelu,
            _ => Activation::SiluGated,
        },
        "gelu_new" | "gelu_pytorch_tanh" | "gelu_tanh" => {
            if matches!(arch, Architecture::Gpt2 | Architecture::Phi) {
                Activation::Gelu
            } else {
                Activation::GeluGated
            }
        }
        _ => arch.default_activation(),
    }
}

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
    /// The model architecture (Llama, Mistral, Gpt2, Phi, Qwen2, etc.).
    #[serde(default)]
    pub architecture: Architecture,
    /// FFN shape/activation (SwiGLU vs GELU vs Gemma's gated GELU).
    #[serde(default)]
    pub activation: Activation,
    /// Norm layer kind (RMSNorm vs LayerNorm).
    #[serde(default)]
    pub norm_type: NormType,
    /// Whether to apply RoPE in attention. False for GPT-2/Phi, which use
    /// learned positional embeddings instead.
    #[serde(default = "default_true")]
    pub use_rope: bool,
    /// Maximum length for learned positional embeddings (GPT-2 `wpe`, Phi
    /// `wpe`). `None` means the model has no positional-embedding table.
    #[serde(default)]
    pub position_embeddings: Option<usize>,
    /// Gemma-style QK-norm: per-head RMSNorm applied to q and k after the
    /// projections and before RoPE.
    #[serde(default)]
    pub qk_norm: bool,
    /// Mistral-style sliding-window attention window size. `None` = full
    /// context (no windowing).
    #[serde(default)]
    pub sliding_window: Option<usize>,
    /// Attention head dimension override (Gemma-2 uses `head_dim` distinct
    /// from `hidden_size / num_heads`). `None` = `hidden_size / num_heads`.
    #[serde(default)]
    pub head_dim: Option<usize>,
}

fn default_true() -> bool {
    true
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
        self.head_dim.unwrap_or(self.hidden_size / self.num_heads)
    }

    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads.unwrap_or(self.num_heads)
    }

    pub fn kv_head_dim(&self) -> usize {
        self.head_dim()
    }

    pub fn num_kv_groups(&self) -> usize {
        self.num_heads / self.num_kv_heads()
    }

    pub fn ff_dim(&self) -> usize {
        self.intermediate_size
    }

    /// Whether the FFN is gated (SwiGLU/GeGLU) vs a single FC + activation.
    pub fn uses_gated_ffn(&self) -> bool {
        self.activation.is_gated()
    }

    pub fn uses_rope(&self) -> bool {
        self.use_rope
    }

    pub fn uses_layer_norm(&self) -> bool {
        self.norm_type == NormType::LayerNorm
    }

    /// The FFN activation convention for this model. Configs parsed from
    /// HF/GGUF metadata set `activation` explicitly; this accessor is the
    /// single source used by the forward path.
    pub fn default_activation(&self) -> Activation {
        self.activation
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
            architecture: Architecture::Llama,
            activation: Activation::SiluGated,
            norm_type: NormType::RmsNorm,
            use_rope: true,
            position_embeddings: None,
            qk_norm: false,
            sliding_window: None,
            head_dim: None,
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
            architecture: Architecture::Llama,
            activation: Activation::SiluGated,
            norm_type: NormType::RmsNorm,
            use_rope: true,
            position_embeddings: None,
            qk_norm: false,
            sliding_window: None,
            head_dim: None,
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

        let architecture = Architecture::from_huggingface(&v).unwrap_or(Architecture::Llama);

        // Per-architecture defaults, overridable by explicit config keys.
        let activation = v
            .get("hidden_act")
            .or_else(|| v.get("hidden_activation"))
            .or_else(|| v.get("activation_function"))
            .and_then(|a| a.as_str())
            .map(|s| parse_activation(s, &architecture))
            .unwrap_or_else(|| architecture.default_activation());

        let norm_type = if v.get("layer_norm_eps").is_some() && !architecture.uses_rms_norm() {
            NormType::LayerNorm
        } else if architecture.uses_rms_norm() {
            NormType::RmsNorm
        } else {
            NormType::LayerNorm
        };

        let use_rope = v.get("use_rope").and_then(|u| u.as_bool()).unwrap_or_else(|| architecture.uses_rope());
        let position_embeddings = if architecture.uses_rope() {
            None
        } else {
            // GPT-2/Phi embed learned positions up to max_position_embeddings.
            v.get("position_embeddings")
                .and_then(|p| p.as_u64())
                .map(|n| n as usize)
                .or(Some(max_seq_len))
        };
        let qk_norm = v.get("qk_norm").and_then(|q| q.as_bool()).unwrap_or_else(|| {
            matches!(architecture, Architecture::Gemma)
        });
        let sliding_window = get_u64("sliding_window").map(Some).unwrap_or(None);
        let head_dim = get_u64("head_dim").map(Some).unwrap_or(None);

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
            architecture,
            activation,
            norm_type,
            use_rope,
            position_embeddings,
            qk_norm,
            sliding_window,
            head_dim,
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

    #[test]
    fn test_architecture_parsing_llama() {
        let json = r#"{
            "architectures": ["LlamaForCausalLM"],
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "intermediate_size": 11008,
            "rms_norm_eps": 1e-5,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0
        }"#;

        let config = ModelConfig::from_huggingface_json(json).unwrap();
        assert_eq!(config.architecture, Architecture::Llama);
    }

    #[test]
    fn test_architecture_parsing_mistral() {
        let json = r#"{
            "architectures": ["MistralForCausalLM"],
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "intermediate_size": 11008,
            "rms_norm_eps": 1e-5,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0
        }"#;

        let config = ModelConfig::from_huggingface_json(json).unwrap();
        assert_eq!(config.architecture, Architecture::Mistral);
    }

    #[test]
    fn test_architecture_parsing_gpt2() {
        let json = r#"{
            "architectures": ["GPT2LMHeadModel"],
            "vocab_size": 50257,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "layer_norm_eps": 1e-5,
            "max_position_embeddings": 1024
        }"#;

        let config = ModelConfig::from_huggingface_json(json).unwrap();
        assert_eq!(config.architecture, Architecture::Gpt2);
    }

    #[test]
    fn test_architecture_parsing_phi() {
        let json = r#"{
            "architectures": ["PhiForCausalLM"],
            "vocab_size": 32000,
            "hidden_size": 2560,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "intermediate_size": 10240,
            "rms_norm_eps": 1e-5,
            "max_position_embeddings": 2048,
            "rope_theta": 1000000.0
        }"#;

        let config = ModelConfig::from_huggingface_json(json).unwrap();
        assert_eq!(config.architecture, Architecture::Phi);
    }

    #[test]
    fn test_architecture_parsing_qwen2() {
        let json = r#"{
            "architectures": ["Qwen2ForCausalLM"],
            "vocab_size": 151936,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "intermediate_size": 11008,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 32768,
            "rope_theta": 1000000.0
        }"#;

        let config = ModelConfig::from_huggingface_json(json).unwrap();
        assert_eq!(config.architecture, Architecture::Qwen2);
    }

    #[test]
    fn test_architecture_parsing_model_type_fallback() {
        let json = r#"{
            "model_type": "mistral",
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "intermediate_size": 11008,
            "rms_norm_eps": 1e-5,
            "max_position_embeddings": 4096
        }"#;

        let config = ModelConfig::from_huggingface_json(json).unwrap();
        assert_eq!(config.architecture, Architecture::Mistral);
    }

    #[test]
    fn test_architecture_from_gguf() {
        assert_eq!(Architecture::from_gguf("llama"), Architecture::Llama);
        assert_eq!(Architecture::from_gguf("mistral"), Architecture::Mistral);
        assert_eq!(Architecture::from_gguf("gpt2"), Architecture::Gpt2);
        assert_eq!(Architecture::from_gguf("phi"), Architecture::Phi);
        assert_eq!(Architecture::from_gguf("gemma"), Architecture::Gemma);
        assert_eq!(Architecture::from_gguf("qwen2"), Architecture::Qwen2);
    }

    #[test]
    fn test_architecture_parsing_gemma() {
        let json = r#"{
            "architectures": ["GemmaForCausalLM"],
            "vocab_size": 256000,
            "hidden_size": 3072,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "intermediate_size": 24576,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 8192,
            "tie_word_embeddings": true,
            "hidden_act": "gelu_pytorch_tanh"
        }"#;

        let config = ModelConfig::from_huggingface_json(json).unwrap();
        assert_eq!(config.architecture, Architecture::Gemma);
        assert_eq!(config.activation, Activation::GeluGated);
        assert_eq!(config.norm_type, NormType::RmsNorm);
        assert!(config.qk_norm, "Gemma uses QK-norm by default");
        assert!(config.use_rope, "Gemma uses RoPE");
        assert!(config.tie_word_embeddings);
    }

    #[test]
    fn test_gpt2_defaults_learned_positions_and_layernorm() {
        let json = r#"{
            "architectures": ["GPT2LMHeadModel"],
            "vocab_size": 50257,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "layer_norm_eps": 1e-5,
            "max_position_embeddings": 1024
        }"#;

        let config = ModelConfig::from_huggingface_json(json).unwrap();
        assert_eq!(config.architecture, Architecture::Gpt2);
        assert!(!config.use_rope, "GPT-2 has no RoPE");
        assert_eq!(config.norm_type, NormType::LayerNorm);
        assert_eq!(config.activation, Activation::Gelu);
        assert_eq!(config.position_embeddings, Some(1024));
    }

    #[test]
    fn test_mistral_sliding_window() {
        let json = r#"{
            "architectures": ["MistralForCausalLM"],
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 14336,
            "rms_norm_eps": 1e-5,
            "max_position_embeddings": 32768,
            "sliding_window": 4096
        }"#;

        let config = ModelConfig::from_huggingface_json(json).unwrap();
        assert_eq!(config.sliding_window, Some(4096));
        assert_eq!(config.num_kv_heads(), 8);
        assert_eq!(config.activation, Activation::SiluGated);
    }

    #[test]
    fn test_gemma2_head_dim_override() {
        let json = r#"{
            "architectures": ["GemmaForCausalLM"],
            "vocab_size": 256000,
            "hidden_size": 3584,
            "num_hidden_layers": 42,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 256,
            "intermediate_size": 14336,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 8192
        }"#;

        let config = ModelConfig::from_huggingface_json(json).unwrap();
        assert_eq!(config.head_dim(), 256);
        assert_eq!(config.head_dim, Some(256));
    }
}
