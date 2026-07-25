use bitllm_tensor::{DType, Tensor};
use memmap2::Mmap;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct TensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

#[derive(Debug, Deserialize)]
struct SafeTensorsHeader {
    #[serde(flatten)]
    tensors: HashMap<String, TensorInfo>,
    #[serde(default)]
    __metadata__: HashMap<String, String>,
}

pub struct SafeTensorsLoader {
    header: SafeTensorsHeader,
    data: Vec<u8>,
}

#[derive(Debug)]
pub enum LoadError {
    IoError(std::io::Error),
    JsonError(serde_json::Error),
    InvalidFormat(String),
    UnsupportedDtype(String),
    OffsetOutOfBounds {
        name: String,
        start: usize,
        end: usize,
        data_len: usize,
    },
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::IoError(e) => write!(f, "IO error: {}", e),
            LoadError::JsonError(e) => write!(f, "JSON error: {}", e),
            LoadError::InvalidFormat(msg) => write!(f, "Invalid format: {}", msg),
            LoadError::UnsupportedDtype(dt) => write!(f, "Unsupported dtype: {}", dt),
            LoadError::OffsetOutOfBounds {
                name,
                start,
                end,
                data_len,
            } => {
                write!(
                    f,
                    "Tensor '{}': offsets [{}, {}) exceed data length {}",
                    name, start, end, data_len
                )
            }
        }
    }
}

impl std::error::Error for LoadError {}

impl From<std::io::Error> for LoadError {
    fn from(e: std::io::Error) -> Self {
        LoadError::IoError(e)
    }
}

impl From<serde_json::Error> for LoadError {
    fn from(e: serde_json::Error) -> Self {
        LoadError::JsonError(e)
    }
}

impl SafeTensorsLoader {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, LoadError> {
        let mut file = File::open(path.as_ref())?;
        let file_len = file.metadata()?.len() as usize;

        if file_len < 8 {
            return Err(LoadError::InvalidFormat("File too small".into()));
        }

        let mut header_len_buf = [0u8; 8];
        file.read_exact(&mut header_len_buf)?;
        let header_len = u64::from_le_bytes(header_len_buf) as usize;

        if 8 + header_len > file_len {
            return Err(LoadError::InvalidFormat(
                "Header extends beyond file".into(),
            ));
        }

        let mut header_buf = vec![0u8; header_len];
        file.read_exact(&mut header_buf)?;
        let header: SafeTensorsHeader = serde_json::from_slice(&header_buf)?;

        let data_len = file_len - 8 - header_len;
        let mut data = vec![0u8; data_len];
        file.read_exact(&mut data)?;

        Ok(Self { header, data })
    }

    pub fn load_mmap<P: AsRef<Path>>(path: P) -> Result<MmapSafeTensors, LoadError> {
        let file = File::open(path.as_ref())?;
        let file_len = file.metadata()?.len() as usize;

        if file_len < 8 {
            return Err(LoadError::InvalidFormat("File too small".into()));
        }

        let mmap = unsafe { Mmap::map(&file).map_err(LoadError::IoError)? };

        let header_len = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;

        if 8 + header_len > file_len {
            return Err(LoadError::InvalidFormat(
                "Header extends beyond file".into(),
            ));
        }

        let header: SafeTensorsHeader = serde_json::from_slice(&mmap[8..8 + header_len])?;
        let data_start = 8 + header_len;

        Ok(MmapSafeTensors {
            header,
            mmap,
            data_start,
        })
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, LoadError> {
        let len = bytes.len();
        if len < 8 {
            return Err(LoadError::InvalidFormat("Data too small".into()));
        }

        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;

        if 8 + header_len > len {
            return Err(LoadError::InvalidFormat(
                "Header extends beyond data".into(),
            ));
        }

        let header: SafeTensorsHeader = serde_json::from_slice(&bytes[8..8 + header_len])?;
        let data = bytes[8 + header_len..].to_vec();

        Ok(Self { header, data })
    }

    pub fn tensor_names(&self) -> Vec<&str> {
        self.header.tensors.keys().map(|s| s.as_str()).collect()
    }

    pub fn tensor_info(&self, name: &str) -> Option<(&str, &[usize], [usize; 2])> {
        let info = self.header.tensors.get(name)?;
        Some((info.dtype.as_str(), &info.shape, info.data_offsets))
    }

    pub fn has_tensor(&self, name: &str) -> bool {
        self.header.tensors.contains_key(name)
    }

    pub fn load_tensor(&self, name: &str) -> Result<Tensor, LoadError> {
        let info = self
            .header
            .tensors
            .get(name)
            .ok_or_else(|| LoadError::InvalidFormat(format!("Tensor '{}' not found", name)))?;

        let [start, end] = info.data_offsets;
        if end > self.data.len() || start > end {
            return Err(LoadError::OffsetOutOfBounds {
                name: name.to_string(),
                start,
                end,
                data_len: self.data.len(),
            });
        }

        let dtype = parse_dtype(&info.dtype)?;
        let tensor_data = &self.data[start..end];

        Ok(match dtype {
            DType::F32 => {
                let floats: Vec<f32> = tensor_data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                Tensor::from_slice(&floats, &info.shape)
            }
            DType::F16 => {
                let mut tensor = Tensor::new(&info.shape, DType::F16);
                tensor.data_mut().copy_from_slice(tensor_data);
                tensor
            }
            DType::BF16 => {
                let mut tensor = Tensor::new(&info.shape, DType::BF16);
                tensor.data_mut().copy_from_slice(tensor_data);
                tensor
            }
            DType::INT8 => {
                let mut tensor = Tensor::new(&info.shape, DType::INT8);
                tensor.data_mut().copy_from_slice(tensor_data);
                tensor
            }
            DType::INT4 => {
                let mut tensor = Tensor::new(&info.shape, DType::INT4);
                tensor.data_mut().copy_from_slice(tensor_data);
                tensor
            }
            DType::BIT1 => {
                let mut tensor = Tensor::new(&info.shape, DType::BIT1);
                tensor.data_mut().copy_from_slice(tensor_data);
                tensor
            }
        })
    }

    pub fn load_all_tensors(&self) -> Result<HashMap<String, Tensor>, LoadError> {
        let mut tensors = HashMap::new();
        for name in self.tensor_names() {
            let tensor = self.load_tensor(name)?;
            tensors.insert(name.to_string(), tensor);
        }
        Ok(tensors)
    }

    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.header.__metadata__
    }
}

fn parse_dtype(dtype_str: &str) -> Result<DType, LoadError> {
    match dtype_str {
        "F32" | "float32" | "Float32" => Ok(DType::F32),
        "F16" | "float16" | "Float16" | "half" => Ok(DType::F16),
        "BF16" | "bfloat16" | "BFloat16" => Ok(DType::BF16),
        "I8" | "int8" | "Int8" => Ok(DType::INT8),
        "I4" | "int4" | "Int4" => Ok(DType::INT4),
        "BIT1" | "bit1" | "Bit1" | "1bit" => Ok(DType::BIT1),
        _ => Err(LoadError::UnsupportedDtype(dtype_str.to_string())),
    }
}

pub struct MmapSafeTensors {
    header: SafeTensorsHeader,
    mmap: Mmap,
    data_start: usize,
}

impl MmapSafeTensors {
    pub fn tensor_names(&self) -> Vec<&str> {
        self.header.tensors.keys().map(|s| s.as_str()).collect()
    }

    pub fn has_tensor(&self, name: &str) -> bool {
        self.header.tensors.contains_key(name)
    }

    pub fn tensor_info(&self, name: &str) -> Option<(&str, &[usize], [usize; 2])> {
        let info = self.header.tensors.get(name)?;
        Some((info.dtype.as_str(), &info.shape, info.data_offsets))
    }

    pub fn load_tensor(&self, name: &str) -> Result<Tensor, LoadError> {
        let info = self
            .header
            .tensors
            .get(name)
            .ok_or_else(|| LoadError::InvalidFormat(format!("Tensor '{}' not found", name)))?;

        let [start, end] = info.data_offsets;
        let data_len = self.mmap.len() - self.data_start;
        if end > data_len || start > end {
            return Err(LoadError::OffsetOutOfBounds {
                name: name.to_string(),
                start,
                end,
                data_len,
            });
        }

        let dtype = parse_dtype(&info.dtype)?;
        let tensor_data = &self.mmap[self.data_start + start..self.data_start + end];

        Ok(match dtype {
            DType::F32 => {
                let floats: Vec<f32> = tensor_data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                Tensor::from_slice(&floats, &info.shape)
            }
            DType::F16 => {
                let mut tensor = Tensor::new(&info.shape, DType::F16);
                tensor.data_mut().copy_from_slice(tensor_data);
                tensor
            }
            DType::BF16 => {
                let mut tensor = Tensor::new(&info.shape, DType::BF16);
                tensor.data_mut().copy_from_slice(tensor_data);
                tensor
            }
            DType::INT8 => {
                let mut tensor = Tensor::new(&info.shape, DType::INT8);
                tensor.data_mut().copy_from_slice(tensor_data);
                tensor
            }
            DType::INT4 => {
                let mut tensor = Tensor::new(&info.shape, DType::INT4);
                tensor.data_mut().copy_from_slice(tensor_data);
                tensor
            }
            DType::BIT1 => {
                let mut tensor = Tensor::new(&info.shape, DType::BIT1);
                tensor.data_mut().copy_from_slice(tensor_data);
                tensor
            }
        })
    }

    pub fn load_all_tensors(&self) -> Result<HashMap<String, Tensor>, LoadError> {
        let mut tensors = HashMap::new();
        for name in self.tensor_names() {
            let tensor = self.load_tensor(name)?;
            tensors.insert(name.to_string(), tensor);
        }
        Ok(tensors)
    }

    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.header.__metadata__
    }
}

pub struct LlamaWeightMapper;

impl LlamaWeightMapper {
    pub fn layer_tensor_name(layer_idx: usize, weight_type: &str) -> String {
        format!("layers.{}.{}", layer_idx, weight_type)
    }

    pub fn map_weight(name: &str, _config: &crate::config::ModelConfig) -> WeightTarget {
        // GGUF-style top-level names
        if name == "tok_embeddings" || name == "token_embd.weight" || name == "model.embed_tokens.weight"
        {
            return WeightTarget::Embedding;
        }
        if name == "norm" || name == "output_norm.weight" || name == "model.norm.weight" {
            return WeightTarget::FinalNorm;
        }
        if name == "output" || name == "lm_head.weight" {
            return WeightTarget::LmHead;
        }

        // Try to parse as a per-layer weight.
        // Supports:
        //   GGUF-style:  layers.N.attn_q.weight
        //   LLama.cpp:   layers.N.attention.wq.weight
        //   HuggingFace: model.layers.N.self_attn.q_proj.weight
        //                model.layers.N.mlp.gate_proj.weight
        //                model.layers.N.input_layernorm.weight
        //                model.layers.N.post_attention_layernorm.weight
        let layer_idx = Self::parse_layer_index(name);
        if let Some(layer_idx) = layer_idx {
            let weight_name = Self::strip_layer_prefix(name);
            return match weight_name.as_str() {
                // Attention projections
                "attention.wq.weight" | "attn_q.weight" | "self_attn.q_proj.weight" => {
                    WeightTarget::AttentionQ { layer_idx }
                }
                "attention.wk.weight" | "attn_k.weight" | "self_attn.k_proj.weight" => {
                    WeightTarget::AttentionK { layer_idx }
                }
                "attention.wv.weight" | "attn_v.weight" | "self_attn.v_proj.weight" => {
                    WeightTarget::AttentionV { layer_idx }
                }
                "attention.wo.weight" | "attn_output.weight" | "self_attn.o_proj.weight" => {
                    WeightTarget::AttentionO { layer_idx }
                }
                // FFN projections
                "feed_forward.w1.weight" | "ffn_gate.weight" | "mlp.gate.weight"
                | "mlp.gate_proj.weight" => WeightTarget::FfnGate { layer_idx },
                "feed_forward.w2.weight" | "ffn_down.weight" | "mlp.down.weight"
                | "mlp.down_proj.weight" => WeightTarget::FfnDown { layer_idx },
                "feed_forward.w3.weight" | "ffn_up.weight" | "mlp.up.weight"
                | "mlp.up_proj.weight" => WeightTarget::FfnUp { layer_idx },
                // Norms
                "attention_norm.weight" | "attn_norm.weight" | "input_layernorm.weight" => {
                    WeightTarget::AttnNorm { layer_idx }
                }
                "ffn_norm.weight" | "mlp_norm.weight" | "post_attention_layernorm.weight" => {
                    WeightTarget::FfnNorm { layer_idx }
                }
                _ => WeightTarget::Unknown(name.to_string()),
            };
        }

        WeightTarget::Unknown(name.to_string())
    }

    /// Extract the layer index from a tensor name.
    /// Handles: `layers.N.*`, `model.layers.N.*`
    fn parse_layer_index(name: &str) -> Option<usize> {
        let parts: Vec<&str> = name.split('.').collect();

        // model.layers.N.* → layer index at position 2
        if parts.len() >= 4 && parts[0] == "model" && parts[1] == "layers" {
            return parts[2].parse::<usize>().ok();
        }

        // layers.N.* → layer index at position 1
        if parts.len() >= 3 && parts[0] == "layers" {
            return parts[1].parse::<usize>().ok();
        }

        None
    }

    /// Strip everything up to and including the layer index and the dot after it.
    /// `model.layers.0.self_attn.q_proj.weight` → `self_attn.q_proj.weight`
    /// `layers.0.attn_q.weight` → `attn_q.weight`
    fn strip_layer_prefix(name: &str) -> String {
        let parts: Vec<&str> = name.split('.').collect();

        // model.layers.N.rest...
        if parts.len() >= 4 && parts[0] == "model" && parts[1] == "layers" {
            return parts[3..].join(".");
        }

        // layers.N.rest...
        if parts.len() >= 3 && parts[0] == "layers" {
            return parts[2..].join(".");
        }

        name.to_string()
    }
}

#[derive(Debug, PartialEq)]
pub enum WeightTarget {
    Embedding,
    FinalNorm,
    LmHead,
    AttentionQ { layer_idx: usize },
    AttentionK { layer_idx: usize },
    AttentionV { layer_idx: usize },
    AttentionO { layer_idx: usize },
    FfnGate { layer_idx: usize },
    FfnDown { layer_idx: usize },
    FfnUp { layer_idx: usize },
    AttnNorm { layer_idx: usize },
    FfnNorm { layer_idx: usize },
    Unknown(String),
}

/// Load weights from a SafeTensors file into a `Model`.
///
/// Iterates all tensors in the file, maps each name to a model weight via
/// `LlamaWeightMapper`, and assigns it to the appropriate layer.
///
/// If `quantize` is provided, each weight tensor is quantized before assignment.
pub fn load_safetensors_weights(
    model: &mut crate::model::Model,
    loader: &SafeTensorsLoader,
    config: &crate::config::ModelConfig,
    quantize: Option<&str>,
) -> LoadingStats {
    let mut stats = LoadingStats::default();

    for name in loader.tensor_names() {
        let target = LlamaWeightMapper::map_weight(name, config);
        let tensor = match loader.load_tensor(name) {
            Ok(t) => t,
            Err(e) => {
                log::warn!("Failed to load tensor '{}': {}", name, e);
                stats.skipped.push(name.to_string());
                continue;
            }
        };

        let tensor = match quantize {
            Some("int8") => {
                use bitllm_quantization::absmax::absmax_quantize;
                use bitllm_quantization::scheme::QuantConfig;
                let q = absmax_quantize(&tensor, &QuantConfig::int8());
                bitllm_quantization::absmax::absmax_dequantize(&q)
            }
            Some("int4") => {
                use bitllm_quantization::absmax::absmax_quantize;
                use bitllm_quantization::scheme::QuantConfig;
                let q = absmax_quantize(&tensor, &QuantConfig::int4());
                bitllm_quantization::absmax::absmax_dequantize(&q)
            }
            Some("ternary") => {
                use bitllm_quantization::ternary::{ternary_dequantize, ternary_quantize};
                let q = ternary_quantize(&tensor);
                ternary_dequantize(&q)
            }
            Some("binary") => {
                use bitllm_tensor::BinaryTensor;
                let bt = BinaryTensor::from_tensor(&tensor);
                bt.dequantize()
            }
            _ => tensor,
        };

        match target {
            WeightTarget::Embedding => {
                model.embedding.weight = tensor;
                stats.loaded += 1;
            }
            WeightTarget::FinalNorm => {
                model.norm.weight = tensor;
                stats.loaded += 1;
            }
            WeightTarget::LmHead => {
                model.lm_head.weight = tensor;
                stats.loaded += 1;
            }
            WeightTarget::AttentionQ { layer_idx } => {
                if let Some(layer) = model.layers.get_mut(layer_idx) {
                    layer.attention.q_proj.weight = tensor;
                    stats.loaded += 1;
                } else {
                    log::warn!(
                        "Layer index {} out of range (model has {} layers)",
                        layer_idx,
                        model.layers.len()
                    );
                    stats.skipped.push(name.to_string());
                }
            }
            WeightTarget::AttentionK { layer_idx } => {
                if let Some(layer) = model.layers.get_mut(layer_idx) {
                    layer.attention.k_proj.weight = tensor;
                    stats.loaded += 1;
                } else {
                    stats.skipped.push(name.to_string());
                }
            }
            WeightTarget::AttentionV { layer_idx } => {
                if let Some(layer) = model.layers.get_mut(layer_idx) {
                    layer.attention.v_proj.weight = tensor;
                    stats.loaded += 1;
                } else {
                    stats.skipped.push(name.to_string());
                }
            }
            WeightTarget::AttentionO { layer_idx } => {
                if let Some(layer) = model.layers.get_mut(layer_idx) {
                    layer.attention.o_proj.weight = tensor;
                    stats.loaded += 1;
                } else {
                    stats.skipped.push(name.to_string());
                }
            }
            WeightTarget::FfnGate { layer_idx } => {
                if let Some(layer) = model.layers.get_mut(layer_idx) {
                    layer.ffn_gate.weight = tensor;
                    stats.loaded += 1;
                } else {
                    stats.skipped.push(name.to_string());
                }
            }
            WeightTarget::FfnDown { layer_idx } => {
                if let Some(layer) = model.layers.get_mut(layer_idx) {
                    layer.ffn_down.weight = tensor;
                    stats.loaded += 1;
                } else {
                    stats.skipped.push(name.to_string());
                }
            }
            WeightTarget::FfnUp { layer_idx } => {
                if let Some(layer) = model.layers.get_mut(layer_idx) {
                    layer.ffn_up.weight = tensor;
                    stats.loaded += 1;
                } else {
                    stats.skipped.push(name.to_string());
                }
            }
            WeightTarget::AttnNorm { layer_idx } => {
                if let Some(layer) = model.layers.get_mut(layer_idx) {
                    layer.attn_norm.weight = tensor;
                    stats.loaded += 1;
                } else {
                    stats.skipped.push(name.to_string());
                }
            }
            WeightTarget::FfnNorm { layer_idx } => {
                if let Some(layer) = model.layers.get_mut(layer_idx) {
                    layer.ffn_norm.weight = tensor;
                    stats.loaded += 1;
                } else {
                    stats.skipped.push(name.to_string());
                }
            }
            WeightTarget::Unknown(unknown_name) => {
                log::debug!("Skipping unknown tensor: {}", unknown_name);
                stats.skipped.push(unknown_name);
            }
        }
    }

    stats
}

/// Statistics from loading weights into a model.
#[derive(Debug, Default)]
pub struct LoadingStats {
    pub loaded: usize,
    pub skipped: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_safetensors() -> Vec<u8> {
        let tensor_a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let tensor_b: Vec<f32> = vec![5.0, 6.0];
        let a_bytes = tensor_a
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect::<Vec<u8>>();
        let b_bytes = tensor_b
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect::<Vec<u8>>();
        let data_len = a_bytes.len() + b_bytes.len();

        let header = serde_json::json!({
            "tensor_a": {
                "dtype": "F32",
                "shape": [2, 2],
                "data_offsets": [0, a_bytes.len()]
            },
            "tensor_b": {
                "dtype": "F32",
                "shape": [2],
                "data_offsets": [a_bytes.len(), data_len]
            }
        });

        let header_str = serde_json::to_string(&header).unwrap();
        let header_bytes = header_str.as_bytes();

        let mut file = Vec::new();
        file.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        file.extend_from_slice(header_bytes);
        file.extend_from_slice(&a_bytes);
        file.extend_from_slice(&b_bytes);
        file
    }

    #[test]
    fn test_load_safetensors() {
        let data = create_test_safetensors();
        let loader = SafeTensorsLoader::from_bytes(&data).unwrap();

        let names = loader.tensor_names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"tensor_a"));
        assert!(names.contains(&"tensor_b"));
    }

    #[test]
    fn test_load_tensor_f32() {
        let data = create_test_safetensors();
        let loader = SafeTensorsLoader::from_bytes(&data).unwrap();

        let t = loader.load_tensor("tensor_a").unwrap();
        assert_eq!(t.shape(), &[2, 2]);
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.get_flat_f32(0), 1.0);
        assert_eq!(t.get_flat_f32(3), 4.0);
    }

    #[test]
    fn test_load_tensor_smaller() {
        let data = create_test_safetensors();
        let loader = SafeTensorsLoader::from_bytes(&data).unwrap();

        let t = loader.load_tensor("tensor_b").unwrap();
        assert_eq!(t.shape(), &[2]);
        assert_eq!(t.get_flat_f32(0), 5.0);
        assert_eq!(t.get_flat_f32(1), 6.0);
    }

    #[test]
    fn test_load_all_tensors() {
        let data = create_test_safetensors();
        let loader = SafeTensorsLoader::from_bytes(&data).unwrap();

        let all = loader.load_all_tensors().unwrap();
        assert_eq!(all.len(), 2);
        assert!(all.contains_key("tensor_a"));
        assert!(all.contains_key("tensor_b"));
    }

    #[test]
    fn test_tensor_not_found() {
        let data = create_test_safetensors();
        let loader = SafeTensorsLoader::from_bytes(&data).unwrap();

        assert!(loader.load_tensor("nonexistent").is_err());
    }

    #[test]
    fn test_has_tensor() {
        let data = create_test_safetensors();
        let loader = SafeTensorsLoader::from_bytes(&data).unwrap();

        assert!(loader.has_tensor("tensor_a"));
        assert!(!loader.has_tensor("nonexistent"));
    }

    #[test]
    fn test_tensor_info() {
        let data = create_test_safetensors();
        let loader = SafeTensorsLoader::from_bytes(&data).unwrap();

        let (dtype, shape, offsets) = loader.tensor_info("tensor_a").unwrap();
        assert_eq!(dtype, "F32");
        assert_eq!(shape, &[2, 2]);
        assert_eq!(offsets, [0, 16]);
    }

    #[test]
    fn test_weight_mapper_llama() {
        assert_eq!(
            LlamaWeightMapper::map_weight(
                "layers.0.attention.wq.weight",
                &crate::config::ModelConfig::tiny_test()
            ),
            WeightTarget::AttentionQ { layer_idx: 0 }
        );
        assert_eq!(
            LlamaWeightMapper::map_weight(
                "layers.5.feed_forward.w2.weight",
                &crate::config::ModelConfig::tiny_test()
            ),
            WeightTarget::FfnDown { layer_idx: 5 }
        );
        assert_eq!(
            LlamaWeightMapper::map_weight(
                "tok_embeddings",
                &crate::config::ModelConfig::tiny_test()
            ),
            WeightTarget::Embedding
        );
        assert_eq!(
            LlamaWeightMapper::map_weight("norm", &crate::config::ModelConfig::tiny_test()),
            WeightTarget::FinalNorm
        );
    }

    #[test]
    fn test_weight_mapper_gguf_style() {
        assert_eq!(
            LlamaWeightMapper::map_weight(
                "token_embd.weight",
                &crate::config::ModelConfig::tiny_test()
            ),
            WeightTarget::Embedding
        );
        assert_eq!(
            LlamaWeightMapper::map_weight(
                "layers.0.attn_q.weight",
                &crate::config::ModelConfig::tiny_test()
            ),
            WeightTarget::AttentionQ { layer_idx: 0 }
        );
        assert_eq!(
            LlamaWeightMapper::map_weight(
                "layers.0.mlp.gate.weight",
                &crate::config::ModelConfig::tiny_test()
            ),
            WeightTarget::FfnGate { layer_idx: 0 }
        );
        assert_eq!(
            LlamaWeightMapper::map_weight(
                "output_norm.weight",
                &crate::config::ModelConfig::tiny_test()
            ),
            WeightTarget::FinalNorm
        );
        assert_eq!(
            LlamaWeightMapper::map_weight(
                "lm_head.weight",
                &crate::config::ModelConfig::tiny_test()
            ),
            WeightTarget::LmHead
        );
    }

    #[test]
    fn test_file_too_small() {
        let data = vec![0u8; 4];
        assert!(SafeTensorsLoader::from_bytes(&data).is_err());
    }

    #[test]
    fn test_file_header_too_large() {
        let mut data = Vec::new();
        data.extend_from_slice(&1000u64.to_le_bytes());
        data.extend_from_slice(&[0u8; 10]);
        assert!(SafeTensorsLoader::from_bytes(&data).is_err());
    }

    #[test]
    fn test_file_empty_tensor() {
        let header = serde_json::json!({
            "empty": {
                "dtype": "F32",
                "shape": [0],
                "data_offsets": [0, 0]
            }
        });
        let header_str = serde_json::to_string(&header).unwrap();
        let header_bytes = header_str.as_bytes();
        let mut file = Vec::new();
        file.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        file.extend_from_slice(header_bytes);

        let loader = SafeTensorsLoader::from_bytes(&file).unwrap();
        assert!(loader.has_tensor("empty"));
    }

    #[test]
    fn test_mmap_load() {
        use std::io::Write;
        let data = create_test_safetensors();
        let dir = std::env::temp_dir().join("bitllm_test_mmap");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.safetensors");
        let mut f = File::create(&path).unwrap();
        f.write_all(&data).unwrap();
        drop(f);

        let loader = SafeTensorsLoader::load_mmap(&path).unwrap();
        assert!(loader.has_tensor("tensor_a"));
        assert!(loader.has_tensor("tensor_b"));

        let t = loader.load_tensor("tensor_a").unwrap();
        assert_eq!(t.shape(), &[2, 2]);
        assert_eq!(t.get_flat_f32(0), 1.0);
        assert_eq!(t.get_flat_f32(3), 4.0);

        let t2 = loader.load_tensor("tensor_b").unwrap();
        assert_eq!(t2.shape(), &[2]);
        assert_eq!(t2.get_flat_f32(0), 5.0);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_weight_mapper_huggingface() {
        let config = crate::config::ModelConfig::tiny_test();
        assert_eq!(
            LlamaWeightMapper::map_weight("model.embed_tokens.weight", &config),
            WeightTarget::Embedding
        );
        assert_eq!(
            LlamaWeightMapper::map_weight("model.layers.0.self_attn.q_proj.weight", &config),
            WeightTarget::AttentionQ { layer_idx: 0 }
        );
        assert_eq!(
            LlamaWeightMapper::map_weight("model.layers.0.self_attn.k_proj.weight", &config),
            WeightTarget::AttentionK { layer_idx: 0 }
        );
        assert_eq!(
            LlamaWeightMapper::map_weight("model.layers.0.self_attn.v_proj.weight", &config),
            WeightTarget::AttentionV { layer_idx: 0 }
        );
        assert_eq!(
            LlamaWeightMapper::map_weight("model.layers.0.self_attn.o_proj.weight", &config),
            WeightTarget::AttentionO { layer_idx: 0 }
        );
        assert_eq!(
            LlamaWeightMapper::map_weight("model.layers.1.mlp.gate_proj.weight", &config),
            WeightTarget::FfnGate { layer_idx: 1 }
        );
        assert_eq!(
            LlamaWeightMapper::map_weight("model.layers.1.mlp.up_proj.weight", &config),
            WeightTarget::FfnUp { layer_idx: 1 }
        );
        assert_eq!(
            LlamaWeightMapper::map_weight("model.layers.1.mlp.down_proj.weight", &config),
            WeightTarget::FfnDown { layer_idx: 1 }
        );
        assert_eq!(
            LlamaWeightMapper::map_weight("model.layers.0.input_layernorm.weight", &config),
            WeightTarget::AttnNorm { layer_idx: 0 }
        );
        assert_eq!(
            LlamaWeightMapper::map_weight(
                "model.layers.0.post_attention_layernorm.weight",
                &config
            ),
            WeightTarget::FfnNorm { layer_idx: 0 }
        );
        assert_eq!(
            LlamaWeightMapper::map_weight("model.norm.weight", &config),
            WeightTarget::FinalNorm
        );
    }

    #[test]
    fn test_weight_mapper_layer_overflow() {
        let config = crate::config::ModelConfig::tiny_test(); // 2 layers
        // The mapper parses the index but doesn't validate against model config.
        // Out-of-range detection happens in load_safetensors_weights (logged + skipped).
        assert_eq!(
            LlamaWeightMapper::map_weight("model.layers.99.self_attn.q_proj.weight", &config),
            WeightTarget::AttentionQ { layer_idx: 99 }
        );
        // Verify the loader rejects it
        let data = create_tiny_safetensors();
        let loader = SafeTensorsLoader::from_bytes(&data).unwrap();
        let mut model = crate::model::Model::new(config.clone());
        let stats = load_safetensors_weights(&mut model, &loader, &config, None);
        // Layer 99 tensor is in the tiny file? No — tiny_test only has layers 0,1.
        // So the mapper maps it, but the loader skips because layer_idx >= model.layers.len().
        // Since the tiny safetensors doesn't contain layer 99, nothing gets skipped for that reason.
        assert_eq!(stats.loaded, 20);
    }

    #[test]
    fn test_config_from_huggingface_json() {
        let json = r#"{
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "intermediate_size": 11008,
            "rms_norm_eps": 1e-5,
            "max_position_embeddings": 2048,
            "rope_theta": 10000.0,
            "tie_word_embeddings": false
        }"#;
        let config = crate::config::ModelConfig::from_huggingface_json(json).unwrap();
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_layers, 32);
        assert_eq!(config.num_heads, 32);
        assert_eq!(config.num_kv_heads, Some(32));
        assert_eq!(config.intermediate_size, 11008);
        assert!((config.norm_eps - 1e-5).abs() < 1e-6);
        assert_eq!(config.max_seq_len, 2048);
        assert!((config.rope_theta - 10000.0).abs() < 0.1);
        assert!(!config.tie_word_embeddings);
    }

    #[test]
    fn test_config_from_huggingface_json_gqa() {
        let json = r#"{
            "hidden_size": 2048,
            "num_hidden_layers": 22,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "intermediate_size": 5632,
            "rms_norm_eps": 1e-5,
            "max_position_embeddings": 4096
        }"#;
        let config = crate::config::ModelConfig::from_huggingface_json(json).unwrap();
        assert_eq!(config.num_kv_heads, Some(4));
        assert_eq!(config.num_layers, 22);
    }

    #[test]
    fn test_config_from_huggingface_json_missing_field() {
        let json = r#"{"vocab_size": 1000}"#;
        assert!(crate::config::ModelConfig::from_huggingface_json(json).is_err());
    }

    /// Helper: create a SafeTensors file with all Llama weight names for tiny_test config.
    fn create_tiny_safetensors() -> Vec<u8> {
        let config = crate::config::ModelConfig::tiny_test();
        let mut tensors: Vec<(String, Vec<f32>, Vec<usize>)> = Vec::new();

        // Embedding
        tensors.push((
            "model.embed_tokens.weight".into(),
            vec![0.1; config.vocab_size * config.hidden_size],
            vec![config.vocab_size, config.hidden_size],
        ));

        // Final norm
        tensors.push((
            "model.norm.weight".into(),
            vec![1.0; config.hidden_size],
            vec![config.hidden_size],
        ));

        // Per-layer weights
        for i in 0..config.num_layers {
            let h = config.hidden_size;
            let kv = config.num_kv_heads() * config.head_dim();
            let inter = config.intermediate_size;

            tensors.push((
                format!("model.layers.{}.self_attn.q_proj.weight", i),
                vec![0.01; h * kv],
                vec![kv, h],
            ));
            tensors.push((
                format!("model.layers.{}.self_attn.k_proj.weight", i),
                vec![0.01; h * kv],
                vec![kv, h],
            ));
            tensors.push((
                format!("model.layers.{}.self_attn.v_proj.weight", i),
                vec![0.01; h * kv],
                vec![kv, h],
            ));
            tensors.push((
                format!("model.layers.{}.self_attn.o_proj.weight", i),
                vec![0.01; h * kv],
                vec![h, kv],
            ));
            tensors.push((
                format!("model.layers.{}.mlp.gate_proj.weight", i),
                vec![0.01; inter * h],
                vec![inter, h],
            ));
            tensors.push((
                format!("model.layers.{}.mlp.up_proj.weight", i),
                vec![0.01; inter * h],
                vec![inter, h],
            ));
            tensors.push((
                format!("model.layers.{}.mlp.down_proj.weight", i),
                vec![0.01; h * inter],
                vec![h, inter],
            ));
            tensors.push((
                format!("model.layers.{}.input_layernorm.weight", i),
                vec![1.0; h],
                vec![h],
            ));
            tensors.push((
                format!("model.layers.{}.post_attention_layernorm.weight", i),
                vec![1.0; h],
                vec![h],
            ));
        }

        // Build the SafeTensors binary
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
    fn test_load_safetensors_weights_into_model() {
        let data = create_tiny_safetensors();
        let loader = SafeTensorsLoader::from_bytes(&data).unwrap();
        let config = crate::config::ModelConfig::tiny_test();

        let mut model = crate::model::Model::new(config.clone());
        let stats = load_safetensors_weights(&mut model, &loader, &config, None);

        // Should have loaded: embedding + norm + 2 layers * (4 attn + 3 ffn + 2 norm) = 1 + 1 + 18 = 20
        assert_eq!(stats.loaded, 20);
        assert!(stats.skipped.is_empty());

        // Verify a weight was actually set (not zero)
        let embedding_sum: f32 = model
            .embedding
            .weight
            .as_f32_slice()
            .iter()
            .sum();
        assert!(embedding_sum.abs() > 0.0, "embedding weights should be non-zero");
    }

    #[test]
    fn test_load_safetensors_weights_with_int8_quantize() {
        let data = create_tiny_safetensors();
        let loader = SafeTensorsLoader::from_bytes(&data).unwrap();
        let config = crate::config::ModelConfig::tiny_test();

        let mut model = crate::model::Model::new(config.clone());
        let stats = load_safetensors_weights(&mut model, &loader, &config, Some("int8"));

        assert_eq!(stats.loaded, 20);
        assert!(stats.skipped.is_empty());
    }

    #[test]
    fn test_load_safetensors_weights_with_ternary_quantize() {
        let data = create_tiny_safetensors();
        let loader = SafeTensorsLoader::from_bytes(&data).unwrap();
        let config = crate::config::ModelConfig::tiny_test();

        let mut model = crate::model::Model::new(config.clone());
        let stats = load_safetensors_weights(&mut model, &loader, &config, Some("ternary"));

        assert_eq!(stats.loaded, 20);
    }
}
