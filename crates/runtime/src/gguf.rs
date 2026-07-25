use crate::config::ModelConfig;
use bitllm_tensor::{DType, Tensor};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::path::Path;

const GGUF_MAGIC: u32 = 0x46554747;
const GGUF_DEFAULT_ALIGNMENT: u64 = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    BF16 = 30,
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0),
            7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0),
            9 => Some(Self::Q8_1),
            10 => Some(Self::Q2K),
            11 => Some(Self::Q3K),
            12 => Some(Self::Q4K),
            13 => Some(Self::Q5K),
            14 => Some(Self::Q6K),
            15 => Some(Self::Q8K),
            24 => Some(Self::I8),
            25 => Some(Self::I16),
            26 => Some(Self::I32),
            30 => Some(Self::BF16),
            _ => None,
        }
    }

    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 | Self::I16 => 2,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 => 32,
            Self::Q8_0 | Self::Q8_1 | Self::I8 => 16,
            Self::Q2K => 32,
            Self::Q3K => 32,
            Self::Q4K => 32,
            Self::Q5K => 32,
            Self::Q6K => 32,
            Self::Q8K => 32,
            Self::I32 => 4,
        }
    }

    pub fn to_dtype(&self) -> Option<DType> {
        match self {
            Self::F32 => Some(DType::F32),
            Self::F16 => Some(DType::F16),
            Self::BF16 => Some(DType::BF16),
            Self::I8 | Self::Q8_0 | Self::Q8_1 | Self::Q8K => Some(DType::INT8),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum MetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<MetadataValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl MetadataValue {
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Uint8(v) => Some(*v as u64),
            Self::Int8(v) => Some(*v as u64),
            Self::Uint16(v) => Some(*v as u64),
            Self::Int16(v) => Some(*v as u64),
            Self::Uint32(v) => Some(*v as u64),
            Self::Int32(v) => Some(*v as u64),
            Self::Uint64(v) => Some(*v),
            Self::Int64(v) => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::Float32(v) => Some(*v),
            Self::Float64(v) => Some(*v as f32),
            Self::Uint32(v) => Some(*v as f32),
            Self::Int32(v) => Some(*v as f32),
            Self::Uint64(v) => Some(*v as f32),
            Self::Int64(v) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[MetadataValue]> {
        match self {
            Self::Array(v) => Some(v),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub struct GgufTensorInfo {
    pub name: String,
    pub n_dimensions: u32,
    pub dimensions: Vec<u64>,
    pub ggml_type: GgmlType,
    pub offset: u64,
}

#[derive(Debug)]
pub struct GgufHeader {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata: HashMap<String, MetadataValue>,
    pub tensor_infos: Vec<GgufTensorInfo>,
    pub alignment: u64,
}

#[derive(Debug)]
pub enum GgufError {
    Io(io::Error),
    InvalidMagic,
    UnsupportedVersion(u32),
    InvalidMetadataType(u32),
    InvalidGgmlType(u32),
    MissingField(String),
}

impl std::fmt::Display for GgufError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {}", e),
            Self::InvalidMagic => write!(f, "Invalid GGUF magic number"),
            Self::UnsupportedVersion(v) => write!(f, "Unsupported GGUF version: {}", v),
            Self::InvalidMetadataType(t) => write!(f, "Invalid metadata value type: {}", t),
            Self::InvalidGgmlType(t) => write!(f, "Invalid GGML type: {}", t),
            Self::MissingField(k) => write!(f, "Missing required field: {}", k),
        }
    }
}

impl std::error::Error for GgufError {}
impl From<io::Error> for GgufError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn read_u8(&mut self) -> io::Result<u8> {
        if self.pos >= self.data.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "overrun"));
        }
        let v = self.data[self.pos];
        self.pos += 1;
        Ok(v)
    }

    fn read_i8(&mut self) -> io::Result<i8> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> io::Result<u16> {
        let mut buf = [0u8; 2];
        self.read_exact(&mut buf)?;
        Ok(u16::from_le_bytes(buf))
    }

    fn read_i16(&mut self) -> io::Result<i16> {
        let mut buf = [0u8; 2];
        self.read_exact(&mut buf)?;
        Ok(i16::from_le_bytes(buf))
    }

    fn read_u32(&mut self) -> io::Result<u32> {
        let mut buf = [0u8; 4];
        self.read_exact(&mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    fn read_i32(&mut self) -> io::Result<i32> {
        let mut buf = [0u8; 4];
        self.read_exact(&mut buf)?;
        Ok(i32::from_le_bytes(buf))
    }

    fn read_u64(&mut self) -> io::Result<u64> {
        let mut buf = [0u8; 8];
        self.read_exact(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }

    fn read_i64(&mut self) -> io::Result<i64> {
        let mut buf = [0u8; 8];
        self.read_exact(&mut buf)?;
        Ok(i64::from_le_bytes(buf))
    }

    fn read_f32(&mut self) -> io::Result<f32> {
        let mut buf = [0u8; 4];
        self.read_exact(&mut buf)?;
        Ok(f32::from_le_bytes(buf))
    }

    fn read_f64(&mut self) -> io::Result<f64> {
        let mut buf = [0u8; 8];
        self.read_exact(&mut buf)?;
        Ok(f64::from_le_bytes(buf))
    }

    fn read_bytes(&mut self, n: usize) -> io::Result<&'a [u8]> {
        if self.pos + n > self.data.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "overrun"));
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> io::Result<()> {
        if self.pos + buf.len() > self.data.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "overrun"));
        }
        buf.copy_from_slice(&self.data[self.pos..self.pos + buf.len()]);
        self.pos += buf.len();
        Ok(())
    }

    fn read_gguf_string(&mut self) -> io::Result<String> {
        let len = self.read_u64()? as usize;
        let bytes = self.read_bytes(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    fn pos(&self) -> usize {
        self.pos
    }
}

fn read_metadata_value(cursor: &mut Cursor, type_id: u32) -> Result<MetadataValue, GgufError> {
    match type_id {
        0 => Ok(MetadataValue::Uint8(cursor.read_u8()?)),
        1 => Ok(MetadataValue::Int8(cursor.read_i8()?)),
        2 => Ok(MetadataValue::Uint16(cursor.read_u16()?)),
        3 => Ok(MetadataValue::Int16(cursor.read_i16()?)),
        4 => Ok(MetadataValue::Uint32(cursor.read_u32()?)),
        5 => Ok(MetadataValue::Int32(cursor.read_i32()?)),
        6 => Ok(MetadataValue::Float32(cursor.read_f32()?)),
        7 => Ok(MetadataValue::Bool(cursor.read_u8()? != 0)),
        8 => Ok(MetadataValue::String(cursor.read_gguf_string()?)),
        9 => {
            let elem_type = cursor.read_u32()?;
            let len = cursor.read_u64()? as usize;
            let mut arr = Vec::with_capacity(len);
            for _ in 0..len {
                arr.push(read_metadata_value(cursor, elem_type)?);
            }
            Ok(MetadataValue::Array(arr))
        }
        10 => Ok(MetadataValue::Uint64(cursor.read_u64()?)),
        11 => Ok(MetadataValue::Int64(cursor.read_i64()?)),
        12 => Ok(MetadataValue::Float64(cursor.read_f64()?)),
        _ => Err(GgufError::InvalidMetadataType(type_id)),
    }
}

fn read_header(cursor: &mut Cursor) -> Result<GgufHeader, GgufError> {
    let magic = cursor.read_u32()?;
    if magic != GGUF_MAGIC {
        return Err(GgufError::InvalidMagic);
    }

    let version = cursor.read_u32()?;
    if !(2..=3).contains(&version) {
        return Err(GgufError::UnsupportedVersion(version));
    }

    let tensor_count = cursor.read_u64()?;
    let metadata_kv_count = cursor.read_u64()?;

    let mut metadata = HashMap::new();
    for _ in 0..metadata_kv_count {
        let key = cursor.read_gguf_string()?;
        let value_type = cursor.read_u32()?;
        let value = read_metadata_value(cursor, value_type)?;
        metadata.insert(key, value);
    }

    let alignment = metadata
        .get("general.alignment")
        .and_then(|v| v.as_u64())
        .unwrap_or(GGUF_DEFAULT_ALIGNMENT);

    let mut tensor_infos = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = cursor.read_gguf_string()?;
        let n_dimensions = cursor.read_u32()?;
        let mut dimensions = Vec::with_capacity(n_dimensions as usize);
        for _ in 0..n_dimensions {
            dimensions.push(cursor.read_u64()?);
        }
        let ggml_type_id = cursor.read_u32()?;
        let ggml_type =
            GgmlType::from_u32(ggml_type_id).ok_or(GgufError::InvalidGgmlType(ggml_type_id))?;
        let offset = cursor.read_u64()?;

        tensor_infos.push(GgufTensorInfo {
            name,
            n_dimensions,
            dimensions,
            ggml_type,
            offset,
        });
    }

    Ok(GgufHeader {
        version,
        tensor_count,
        metadata,
        tensor_infos,
        alignment,
    })
}

fn align_offset(offset: u64, alignment: u64) -> u64 {
    (offset + alignment - 1) & !(alignment - 1)
}

#[derive(Debug)]
pub struct GgufLoader {
    header: GgufHeader,
    mmap: Mmap,
    data_offset: u64,
}

impl GgufLoader {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, GgufError> {
        let file = File::open(path.as_ref())?;
        let mmap = unsafe { Mmap::map(&file)? };

        let mut cursor = Cursor::new(&mmap);
        let header = read_header(&mut cursor)?;

        let data_offset = align_offset(cursor.pos() as u64, header.alignment);

        Ok(Self {
            header,
            mmap,
            data_offset,
        })
    }

    pub fn header(&self) -> &GgufHeader {
        &self.header
    }

    pub fn metadata(&self) -> &HashMap<String, MetadataValue> {
        &self.header.metadata
    }

    pub fn metadata_str(&self, key: &str) -> Option<&str> {
        self.header.metadata.get(key).and_then(|v| v.as_str())
    }

    pub fn metadata_u64(&self, key: &str) -> Option<u64> {
        self.header.metadata.get(key).and_then(|v| v.as_u64())
    }

    pub fn metadata_f32(&self, key: &str) -> Option<f32> {
        self.header.metadata.get(key).and_then(|v| v.as_f32())
    }

    pub fn tensor_names(&self) -> Vec<&str> {
        self.header
            .tensor_infos
            .iter()
            .map(|t| t.name.as_str())
            .collect()
    }

    pub fn tensor_info(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.header.tensor_infos.iter().find(|t| t.name == name)
    }

    pub fn has_tensor(&self, name: &str) -> bool {
        self.header.tensor_infos.iter().any(|t| t.name == name)
    }

    pub fn load_tensor(&self, name: &str) -> Result<Tensor, GgufError> {
        let info = self
            .tensor_info(name)
            .ok_or_else(|| GgufError::MissingField(name.to_string()))?;

        let tensor_data_start = self.data_offset + info.offset;

        match info.ggml_type {
            GgmlType::F32 => {
                let n = info.dimensions.iter().product::<u64>() as usize;
                let bytes = n * 4;
                let data =
                    &self.mmap[tensor_data_start as usize..tensor_data_start as usize + bytes];
                let floats: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                let shape: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
                Ok(Tensor::from_slice(&floats, &shape))
            }
            GgmlType::F16 => {
                let shape: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
                let mut tensor = Tensor::new(&shape, DType::F16);
                let bytes = tensor.data().len();
                tensor.data_mut().copy_from_slice(
                    &self.mmap[tensor_data_start as usize..tensor_data_start as usize + bytes],
                );
                Ok(tensor)
            }
            GgmlType::BF16 => {
                let shape: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
                let mut tensor = Tensor::new(&shape, DType::BF16);
                let bytes = tensor.data().len();
                tensor.data_mut().copy_from_slice(
                    &self.mmap[tensor_data_start as usize..tensor_data_start as usize + bytes],
                );
                Ok(tensor)
            }
            GgmlType::I8 | GgmlType::Q8_0 => {
                let shape: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
                let n = shape.iter().product::<usize>();
                let raw = &self.mmap[tensor_data_start as usize..];
                let floats: Vec<f32> = match info.ggml_type {
                    GgmlType::I8 => (0..n).map(|i| raw[i] as i8 as f32).collect(),
                    GgmlType::Q8_0 => {
                        let blocks = n.div_ceil(16);
                        let mut result = Vec::with_capacity(n);
                        for b in 0..blocks {
                            let base = b * (1 + 16);
                            if base + 17 > raw.len() {
                                break;
                            }
                            let scale = raw[base] as i8 as f32;
                            let block = &raw[base + 1..base + 17];
                            for &byte in block {
                                if result.len() >= n {
                                    break;
                                }
                                let q = byte as i8;
                                result.push(q as f32 * scale);
                            }
                        }
                        result
                    }
                    _ => unreachable!(),
                };
                Ok(Tensor::from_slice(&floats, &shape))
            }
            GgmlType::Q4_0 => {
                let shape: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
                let n = shape.iter().product::<usize>();
                let raw = &self.mmap[tensor_data_start as usize..];
                let blocks = n.div_ceil(32);
                let mut floats = Vec::with_capacity(n);
                for b in 0..blocks {
                    let base = b * (2 + 16);
                    if base + 18 > raw.len() {
                        break;
                    }
                    let d_bytes = [raw[base], raw[base + 1]];
                    let scale = f16_to_f32(u16::from_le_bytes(d_bytes));
                    let data = &raw[base + 2..base + 18];
                    for byte in data {
                        let lo = (byte & 0x0F) as i8 - 8;
                        let hi = ((byte >> 4) & 0x0F) as i8 - 8;
                        floats.push(lo as f32 * scale);
                        floats.push(hi as f32 * scale);
                    }
                }
                floats.truncate(n);
                Ok(Tensor::from_slice(&floats, &shape))
            }
            other => Err(GgufError::InvalidGgmlType(other as u32)),
        }
    }

    pub fn tensor_infos(&self) -> &[GgufTensorInfo] {
        &self.header.tensor_infos
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mantissa = (h & 0x3FF) as u32;

    if exp == 0 {
        if mantissa == 0 {
            f32::from_bits(sign << 31)
        } else {
            let mut m = mantissa;
            let mut e = 1u32;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF;
            f32::from_bits((sign << 31) | ((e + 127 - 15) << 23) | (m << 13))
        }
    } else if exp == 31 {
        f32::from_bits((sign << 31) | 0x7F800000 | (mantissa << 13))
    } else {
        f32::from_bits((sign << 31) | ((exp + 127 - 15) << 23) | (mantissa << 13))
    }
}

impl GgufLoader {
    pub fn config_from_metadata(&self) -> Option<ModelConfig> {
        let arch = self.metadata_str("general.architecture")?;

        let prefix = format!("{}.", arch);
        let meta = self.metadata();

        let get_u64 = |key: &str| -> Option<u64> {
            meta.get(key)
                .or_else(|| meta.get(&format!("{}{}", prefix, key)))
                .and_then(|v| v.as_u64())
        };
        let get_f32 = |key: &str| -> Option<f32> {
            meta.get(key)
                .or_else(|| meta.get(&format!("{}{}", prefix, key)))
                .and_then(|v| v.as_f32())
        };
        let get_bool = |key: &str| -> Option<bool> {
            meta.get(key)
                .or_else(|| meta.get(&format!("{}{}", prefix, key)))
                .and_then(|v| v.as_bool())
        };

        let hidden_size = get_u64("embedding_length")? as usize;
        let num_layers = get_u64("block_count")? as usize;
        let num_heads = get_u64("attention.head_count")? as usize;
        let num_kv_heads = get_u64("attention.head_count_kv").map(|v| v as usize);
        let intermediate_size = get_u64("feed_forward_length")
            .map(|v| v as usize)
            .unwrap_or(hidden_size * 4);
        let max_seq_len = get_u64("context_length").unwrap_or(2048) as usize;
        let norm_eps = get_f32("attention.layer_norm_rms_epsilon").unwrap_or(1e-5);
        let rope_theta = get_f32("rope.freq_base").unwrap_or(10000.0);
        let tie_word_embeddings = get_bool("tie_word_embeddings").unwrap_or(false);

        let vocab_size = meta
            .get("tokenizer.ggml.tokens")
            .and_then(|v: &MetadataValue| v.as_array())
            .map(|arr: &[MetadataValue]| arr.len())
            .unwrap_or(32000);

        Some(ModelConfig {
            vocab_size,
            hidden_size,
            num_layers,
            num_heads,
            num_kv_heads: Some(num_kv_heads.unwrap_or(num_heads)),
            intermediate_size,
            norm_eps,
            max_seq_len,
            rope_theta,
            tie_word_embeddings,
        })
    }
}

pub struct GgufWeightMapper;

impl GgufWeightMapper {
    pub fn map_weight(name: &str) -> crate::loader::WeightTarget {
        if name == "token_embd.weight" {
            return crate::loader::WeightTarget::Embedding;
        }
        if name == "output_norm.weight" {
            return crate::loader::WeightTarget::FinalNorm;
        }
        if name == "output.weight" {
            return crate::loader::WeightTarget::LmHead;
        }

        if let Some(rest) = name.strip_prefix("blk.") {
            let parts: Vec<&str> = rest.split('.').collect();
            if parts.len() >= 2 {
                if let Ok(layer_idx) = parts[0].parse::<usize>() {
                    let weight_name = parts[1..].join(".");
                    return match weight_name.as_str() {
                        "attn_q.weight" => crate::loader::WeightTarget::AttentionQ { layer_idx },
                        "attn_k.weight" => crate::loader::WeightTarget::AttentionK { layer_idx },
                        "attn_v.weight" => crate::loader::WeightTarget::AttentionV { layer_idx },
                        "attn_output.weight" => {
                            crate::loader::WeightTarget::AttentionO { layer_idx }
                        }
                        "ffn_gate.weight" => crate::loader::WeightTarget::FfnGate { layer_idx },
                        "ffn_down.weight" => crate::loader::WeightTarget::FfnDown { layer_idx },
                        "ffn_up.weight" => crate::loader::WeightTarget::FfnUp { layer_idx },
                        "attn_norm.weight" => crate::loader::WeightTarget::AttnNorm { layer_idx },
                        "ffn_norm.weight" => crate::loader::WeightTarget::FfnNorm { layer_idx },
                        _ => crate::loader::WeightTarget::Unknown(name.to_string()),
                    };
                }
            }
        }

        crate::loader::WeightTarget::Unknown(name.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_u8(buf: &mut Vec<u8>, v: u8) {
        buf.push(v);
    }

    fn write_u32(buf: &mut Vec<u8>, v: u32) {
        buf.extend_from_slice(&v.to_le_bytes());
    }

    fn write_u64(buf: &mut Vec<u8>, v: u64) {
        buf.extend_from_slice(&v.to_le_bytes());
    }

    fn write_gguf_string(buf: &mut Vec<u8>, s: &str) {
        write_u64(buf, s.len() as u64);
        buf.extend_from_slice(s.as_bytes());
    }

    fn write_metadata_kv(buf: &mut Vec<u8>, key: &str, value: &MetadataValue) {
        write_gguf_string(buf, key);
        match value {
            MetadataValue::Uint64(v) => {
                write_u32(buf, 10);
                write_u64(buf, *v);
            }
            MetadataValue::Float32(v) => {
                write_u32(buf, 6);
                buf.extend_from_slice(&v.to_le_bytes());
            }
            MetadataValue::String(v) => {
                write_u32(buf, 8);
                write_gguf_string(buf, v);
            }
            MetadataValue::Bool(v) => {
                write_u32(buf, 7);
                write_u8(buf, if *v { 1 } else { 0 });
            }
            _ => panic!("unsupported test value type"),
        }
    }

    fn create_test_gguf(f32_tensors: Vec<(&str, Vec<f32>)>) -> Vec<u8> {
        let mut buf = Vec::new();

        write_u32(&mut buf, GGUF_MAGIC);
        write_u32(&mut buf, 3);

        let tensor_count = f32_tensors.len() as u64;
        let metadata = vec![
            (
                "general.architecture".to_string(),
                MetadataValue::String("llama".to_string()),
            ),
            ("llama.block_count".to_string(), MetadataValue::Uint64(2)),
            (
                "llama.embedding_length".to_string(),
                MetadataValue::Uint64(64),
            ),
            (
                "llama.attention.head_count".to_string(),
                MetadataValue::Uint64(4),
            ),
            ("general.alignment".to_string(), MetadataValue::Uint64(32)),
        ];

        write_u64(&mut buf, tensor_count);
        write_u64(&mut buf, metadata.len() as u64);

        for (key, value) in &metadata {
            write_metadata_kv(&mut buf, key, value);
        }

        let alignment = 32u64;
        let mut current_offset = 0u64;

        for (name, data) in &f32_tensors {
            write_gguf_string(&mut buf, name);
            let n_dims = if data.len() > 1 { 2 } else { 1 };
            write_u32(&mut buf, n_dims);
            if n_dims == 2 {
                let rows = 1u64;
                let cols = data.len() as u64;
                write_u64(&mut buf, rows);
                write_u64(&mut buf, cols);
            } else {
                write_u64(&mut buf, data.len() as u64);
            }
            write_u32(&mut buf, GgmlType::F32 as u32);
            write_u64(&mut buf, current_offset);
            current_offset += (data.len() * 4) as u64;
        }

        let header_end = buf.len() as u64;
        let data_start = align_offset(header_end, alignment);
        let padding = (data_start - header_end) as usize;
        buf.extend(std::iter::repeat(0u8).take(padding));

        for (_name, data) in &f32_tensors {
            for val in data {
                buf.extend_from_slice(&val.to_le_bytes());
            }
        }

        buf
    }

    #[test]
    fn test_gguf_magic_and_version() {
        let data = create_test_gguf(vec![]);
        let loader = GgufLoader::load_from_bytes(&data).unwrap();
        assert_eq!(loader.header().version, 3);
        assert_eq!(loader.header().tensor_count, 0);
    }

    #[test]
    fn test_gguf_metadata() {
        let data = create_test_gguf(vec![]);
        let loader = GgufLoader::load_from_bytes(&data).unwrap();
        assert_eq!(loader.metadata_str("general.architecture"), Some("llama"));
        assert_eq!(loader.metadata_u64("llama.block_count"), Some(2));
        assert_eq!(loader.metadata_u64("llama.embedding_length"), Some(64));
    }

    #[test]
    fn test_gguf_tensor_loading() {
        let data = create_test_gguf(vec![
            ("token_embd.weight", vec![1.0, 2.0, 3.0, 4.0]),
            ("output_norm.weight", vec![0.5, -0.5]),
        ]);
        let loader = GgufLoader::load_from_bytes(&data).unwrap();

        assert!(loader.has_tensor("token_embd.weight"));
        assert!(loader.has_tensor("output_norm.weight"));
        assert!(!loader.has_tensor("nonexistent"));

        let t = loader.load_tensor("token_embd.weight").unwrap();
        assert_eq!(t.shape(), &[1, 4]);
        assert_eq!(t.dtype(), DType::F32);
        assert!((t.get_flat_f32(0) - 1.0).abs() < 1e-6);
        assert!((t.get_flat_f32(3) - 4.0).abs() < 1e-6);

        let t2 = loader.load_tensor("output_norm.weight").unwrap();
        assert_eq!(t2.shape(), &[1, 2]);
        assert!((t2.get_flat_f32(0) - 0.5).abs() < 1e-6);
        assert!((t2.get_flat_f32(1) - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn test_config_from_metadata() {
        let data = create_test_gguf(vec![]);
        let loader = GgufLoader::load_from_bytes(&data).unwrap();
        let config = loader.config_from_metadata().unwrap();
        assert_eq!(config.num_layers, 2);
        assert_eq!(config.hidden_size, 64);
        assert_eq!(config.num_heads, 4);
    }

    #[test]
    fn test_weight_mapper() {
        assert_eq!(
            GgufWeightMapper::map_weight("token_embd.weight"),
            crate::loader::WeightTarget::Embedding
        );
        assert_eq!(
            GgufWeightMapper::map_weight("output_norm.weight"),
            crate::loader::WeightTarget::FinalNorm
        );
        assert_eq!(
            GgufWeightMapper::map_weight("output.weight"),
            crate::loader::WeightTarget::LmHead
        );
        assert_eq!(
            GgufWeightMapper::map_weight("blk.0.attn_q.weight"),
            crate::loader::WeightTarget::AttentionQ { layer_idx: 0 }
        );
        assert_eq!(
            GgufWeightMapper::map_weight("blk.3.ffn_gate.weight"),
            crate::loader::WeightTarget::FfnGate { layer_idx: 3 }
        );
        assert_eq!(
            GgufWeightMapper::map_weight("blk.7.ffn_down.weight"),
            crate::loader::WeightTarget::FfnDown { layer_idx: 7 }
        );
        assert_eq!(
            GgufWeightMapper::map_weight("blk.1.attn_norm.weight"),
            crate::loader::WeightTarget::AttnNorm { layer_idx: 1 }
        );
        assert_eq!(
            GgufWeightMapper::map_weight("blk.5.ffn_norm.weight"),
            crate::loader::WeightTarget::FfnNorm { layer_idx: 5 }
        );
    }

    #[test]
    fn test_invalid_magic() {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(b"FAIL");
        assert!(GgufLoader::load_from_bytes(&data).is_err());
    }

    #[test]
    fn test_tensor_not_found() {
        let data = create_test_gguf(vec![("a.weight", vec![1.0])]);
        let loader = GgufLoader::load_from_bytes(&data).unwrap();
        assert!(loader.load_tensor("nonexistent").is_err());
    }

    #[test]
    fn test_tensor_names() {
        let data = create_test_gguf(vec![("a.weight", vec![1.0]), ("b.weight", vec![2.0])]);
        let loader = GgufLoader::load_from_bytes(&data).unwrap();
        let names = loader.tensor_names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"a.weight"));
        assert!(names.contains(&"b.weight"));
    }

    impl GgufLoader {
        fn load_from_bytes(data: &[u8]) -> Result<Self, GgufError> {
            use std::io::Write;
            let dir = std::env::temp_dir().join("bitllm_gguf_test");
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join(format!("test_{:?}.gguf", std::thread::current().id()));
            let mut f = File::create(&path).unwrap();
            f.write_all(data).unwrap();
            drop(f);

            let result = GgufLoader::load(&path);
            std::fs::remove_file(&path).ok();
            result
        }
    }
}
