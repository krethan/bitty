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
        if name == "tok_embeddings" || name == "token_embd.weight" {
            return WeightTarget::Embedding;
        }
        if name == "norm" || name == "output_norm.weight" {
            return WeightTarget::FinalNorm;
        }
        if name == "output" || name == "lm_head.weight" {
            return WeightTarget::LmHead;
        }

        let parts: Vec<&str> = name.split('.').collect();
        if parts.len() >= 3 && parts[0] == "layers" {
            if let Ok(layer_idx) = parts[1].parse::<usize>() {
                let weight_name = parts[2..].join(".");
                return match weight_name.as_str() {
                    "attention.wq.weight" | "attn_q.weight" => {
                        WeightTarget::AttentionQ { layer_idx }
                    }
                    "attention.wk.weight" | "attn_k.weight" => {
                        WeightTarget::AttentionK { layer_idx }
                    }
                    "attention.wv.weight" | "attn_v.weight" => {
                        WeightTarget::AttentionV { layer_idx }
                    }
                    "attention.wo.weight" | "attn_output.weight" => {
                        WeightTarget::AttentionO { layer_idx }
                    }
                    "feed_forward.w1.weight" | "ffn_gate.weight" | "mlp.gate.weight" => {
                        WeightTarget::FfnGate { layer_idx }
                    }
                    "feed_forward.w2.weight" | "ffn_down.weight" | "mlp.down.weight" => {
                        WeightTarget::FfnDown { layer_idx }
                    }
                    "feed_forward.w3.weight" | "ffn_up.weight" | "mlp.up.weight" => {
                        WeightTarget::FfnUp { layer_idx }
                    }
                    "attention_norm.weight" | "attn_norm.weight" => {
                        WeightTarget::AttnNorm { layer_idx }
                    }
                    "ffn_norm.weight" | "mlp_norm.weight" => WeightTarget::FfnNorm { layer_idx },
                    _ => WeightTarget::Unknown(name.to_string()),
                };
            }
        }

        WeightTarget::Unknown(name.to_string())
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
}
