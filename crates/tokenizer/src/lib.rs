use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TokenizerError {
    #[error("Unknown token: {0}")]
    UnknownToken(String),
    #[error("Invalid vocabulary: {0}")]
    InvalidVocabulary(String),
    #[error("Encoding error: {0}")]
    EncodingError(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
    #[error("Load error: {0}")]
    LoadError(String),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenizerConfig {
    pub vocab_size: usize,
    pub bos_token: Option<u32>,
    pub eos_token: Option<u32>,
    pub unk_token: Option<u32>,
}

pub struct BpeTokenizer {
    vocab: HashMap<String, u32>,
    inv_vocab: HashMap<u32, String>,
    merges: Vec<(String, String)>,
    config: TokenizerConfig,
}

impl BpeTokenizer {
    pub fn new(config: TokenizerConfig) -> Self {
        Self {
            vocab: HashMap::new(),
            inv_vocab: HashMap::new(),
            merges: Vec::new(),
            config,
        }
    }

    pub fn from_vocab_and_merges(
        vocab: HashMap<String, u32>,
        merges: Vec<(String, String)>,
    ) -> Self {
        let inv_vocab: HashMap<u32, String> = vocab.iter().map(|(k, &v)| (v, k.clone())).collect();
        let config = TokenizerConfig {
            vocab_size: vocab.len(),
            ..Default::default()
        };
        Self {
            vocab,
            inv_vocab,
            merges,
            config,
        }
    }

    pub fn from_vocab_and_merges_with_config(
        vocab: HashMap<String, u32>,
        merges: Vec<(String, String)>,
        config: TokenizerConfig,
    ) -> Self {
        let inv_vocab: HashMap<u32, String> = vocab.iter().map(|(k, &v)| (v, k.clone())).collect();
        let config = TokenizerConfig {
            vocab_size: vocab.len(),
            ..config
        };
        Self {
            vocab,
            inv_vocab,
            merges,
            config,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    pub fn config(&self) -> &TokenizerConfig {
        &self.config
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let words = self.regex_split(text);
        let mut tokens = Vec::new();

        for word in words {
            let mut chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();

            for (left, right) in &self.merges {
                let mut new_chars = Vec::new();
                let mut i = 0;
                while i < chars.len() {
                    if i + 1 < chars.len() && chars[i] == *left && chars[i + 1] == *right {
                        new_chars.push(format!("{}{}", left, right));
                        i += 2;
                    } else {
                        new_chars.push(chars[i].clone());
                        i += 1;
                    }
                }
                chars = new_chars;
            }

            for c in chars {
                if let Some(&id) = self.vocab.get(&c) {
                    tokens.push(id);
                } else {
                    for ch in c.chars() {
                        let s = ch.to_string();
                        if let Some(&id) = self.vocab.get(&s) {
                            tokens.push(id);
                        }
                    }
                }
            }
        }

        tokens
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String, TokenizerError> {
        let mut result = String::new();
        for &token in tokens {
            if token == self.config.bos_token.unwrap_or(u32::MAX) {
                continue;
            }
            if token == self.config.eos_token.unwrap_or(u32::MAX) {
                break;
            }
            match self.inv_vocab.get(&token) {
                Some(s) => result.push_str(s),
                None => {
                    return Err(TokenizerError::UnknownToken(format!("token_id={}", token)));
                }
            }
        }
        Ok(result)
    }

    pub fn encode_with_special(&self, text: &str, add_bos: bool, add_eos: bool) -> Vec<u32> {
        let mut tokens = Vec::new();
        if add_bos {
            if let Some(bos) = self.config.bos_token {
                tokens.push(bos);
            }
        }
        tokens.extend(self.encode(text));
        if add_eos {
            if let Some(eos) = self.config.eos_token {
                tokens.push(eos);
            }
        }
        tokens
    }

    fn regex_split(&self, text: &str) -> Vec<String> {
        let mut words = Vec::new();
        let mut current = String::new();

        for ch in text.chars() {
            if ch.is_whitespace() || ch.is_ascii_punctuation() {
                if !current.is_empty() {
                    words.push(current.clone());
                    current.clear();
                }
                words.push(ch.to_string());
            } else {
                current.push(ch);
            }
        }

        if !current.is_empty() {
            words.push(current);
        }

        words
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, TokenizerError> {
        let mut file = File::open(path.as_ref())?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        Self::from_json(&contents)
    }

    pub fn from_json(json: &str) -> Result<Self, TokenizerError> {
        let value: serde_json::Value = serde_json::from_str(json)?;

        let model = value
            .get("model")
            .ok_or_else(|| TokenizerError::LoadError("missing 'model' field".into()))?;

        let model_type = model.get("type").and_then(|v| v.as_str()).unwrap_or("BPE");

        if model_type != "BPE" {
            return Err(TokenizerError::LoadError(format!(
                "unsupported model type: {}",
                model_type
            )));
        }

        let vocab_value = model
            .get("vocab")
            .ok_or_else(|| TokenizerError::LoadError("missing 'model.vocab'".into()))?;

        let mut vocab = HashMap::new();
        match vocab_value {
            serde_json::Value::Object(map) => {
                for (key, val) in map {
                    if let Some(id) = val.as_u64() {
                        vocab.insert(key.clone(), id as u32);
                    }
                }
            }
            _ => {
                return Err(TokenizerError::LoadError(
                    "model.vocab must be an object".into(),
                ))
            }
        }

        let merges_value = model
            .get("merges")
            .ok_or_else(|| TokenizerError::LoadError("missing 'model.merges'".into()))?;

        let mut merges = Vec::new();
        if let Some(arr) = merges_value.as_array() {
            for item in arr {
                if let Some(s) = item.as_str() {
                    let parts: Vec<&str> = s.splitn(2, ' ').collect();
                    if parts.len() == 2 {
                        merges.push((parts[0].to_string(), parts[1].to_string()));
                    }
                }
            }
        }

        let mut bos_token = None;
        let mut eos_token = None;

        if let Some(added) = value.get("added_tokens") {
            if let Some(arr) = added.as_array() {
                for token in arr {
                    let content = token.get("content").and_then(|v| v.as_str()).unwrap_or("");
                    let id = token.get("id").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                    let special = token
                        .get("special")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);

                    if special {
                        if content == "<s>" || content == "[BOS]" {
                            bos_token = Some(id);
                        } else if content == "</s>" || content == "[EOS]" {
                            eos_token = Some(id);
                        }
                    }
                }
            }
        }

        let config = TokenizerConfig {
            vocab_size: vocab.len(),
            bos_token,
            eos_token,
            unk_token: vocab.get("<unk>").copied(),
        };

        Ok(Self::from_vocab_and_merges_with_config(
            vocab, merges, config,
        ))
    }
}

pub struct SimpleTokenizer {
    char_to_id: HashMap<char, u32>,
    id_to_char: HashMap<u32, char>,
}

impl SimpleTokenizer {
    pub fn from_vocab(vocab: &[(char, u32)]) -> Self {
        let char_to_id: HashMap<char, u32> = vocab.iter().cloned().collect();
        let id_to_char: HashMap<u32, char> = vocab.iter().map(|(c, id)| (*id, *c)).collect();
        Self {
            char_to_id,
            id_to_char,
        }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        text.chars()
            .filter_map(|c| self.char_to_id.get(&c).copied())
            .collect()
    }

    pub fn decode(&self, tokens: &[u32]) -> String {
        tokens
            .iter()
            .filter_map(|&id| self.id_to_char.get(&id).copied())
            .collect()
    }

    pub fn vocab_size(&self) -> usize {
        self.char_to_id.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_tokenizer() {
        let vocab = vec![('a', 0), ('b', 1), ('c', 2), ('d', 3), (' ', 4), ('!', 5)];
        let tok = SimpleTokenizer::from_vocab(&vocab);

        let encoded = tok.encode("abc");
        assert_eq!(encoded, vec![0, 1, 2]);

        let decoded = tok.decode(&encoded);
        assert_eq!(decoded, "abc");
    }

    #[test]
    fn test_simple_tokenizer_with_spaces() {
        let vocab = vec![('h', 0), ('i', 1), (' ', 2)];
        let tok = SimpleTokenizer::from_vocab(&vocab);

        let encoded = tok.encode("hi hi");
        assert_eq!(encoded, vec![0, 1, 2, 0, 1]);
    }

    #[test]
    fn test_bpe_basic() {
        let mut vocab = HashMap::new();
        vocab.insert("a".to_string(), 0);
        vocab.insert("b".to_string(), 1);
        vocab.insert("c".to_string(), 2);
        vocab.insert("ab".to_string(), 3);
        vocab.insert("bc".to_string(), 4);
        let merges = vec![
            ("a".to_string(), "b".to_string()),
            ("b".to_string(), "c".to_string()),
        ];
        let tok = BpeTokenizer::from_vocab_and_merges(vocab, merges);

        let encoded = tok.encode("abc");
        assert!(encoded.contains(&3) || encoded.contains(&0));
    }

    #[test]
    fn test_decode_with_eos() {
        let mut vocab = HashMap::new();
        vocab.insert("a".to_string(), 0);
        vocab.insert("b".to_string(), 1);
        vocab.insert("eos".to_string(), 2);
        let config = TokenizerConfig {
            vocab_size: 3,
            eos_token: Some(2),
            ..Default::default()
        };
        let tok = BpeTokenizer::from_vocab_and_merges_with_config(vocab, vec![], config);
        let decoded = tok.decode(&[0, 1, 2, 0]).unwrap();
        assert_eq!(decoded, "ab");
    }

    #[test]
    fn test_encode_with_special() {
        let mut vocab = HashMap::new();
        vocab.insert("x".to_string(), 3);
        let config = TokenizerConfig {
            vocab_size: 4,
            bos_token: Some(1),
            eos_token: Some(2),
            ..Default::default()
        };
        let mut tok = BpeTokenizer::new(config);
        tok.vocab = vocab;

        let tokens = tok.encode_with_special("x", true, true);
        assert_eq!(tokens, vec![1, 3, 2]);
    }

    #[test]
    fn test_regex_split() {
        let tok = BpeTokenizer::new(TokenizerConfig::default());
        let words = tok.regex_split("hello, world!");
        assert_eq!(words, vec!["hello", ",", " ", "world", "!"]);
    }

    #[test]
    fn test_load_from_json() {
        let json = r#"{
            "model": {
                "type": "BPE",
                "vocab": {
                    "hello": 0,
                    "world": 1,
                    "lo": 2,
                    "wor": 3,
                    " </s>": 4
                },
                "merges": ["l o", "lo r", "wor ld", "hello world"]
            },
            "added_tokens": [
                {"id": 4, "content": "</s>", "special": true}
            ]
        }"#;

        let tok = BpeTokenizer::from_json(json).unwrap();
        assert_eq!(tok.vocab_size(), 5);
        assert_eq!(tok.config().eos_token, Some(4));

        let encoded = tok.encode("hello");
        assert!(!encoded.is_empty());
    }

    #[test]
    fn test_load_from_json_with_bos_eos() {
        let json = r#"{
            "model": {
                "type": "BPE",
                "vocab": {"a": 0, "b": 1, "c": 2},
                "merges": []
            },
            "added_tokens": [
                {"id": 3, "content": "<s>", "special": true},
                {"id": 4, "content": "</s>", "special": true}
            ]
        }"#;

        let tok = BpeTokenizer::from_json(json).unwrap();
        assert_eq!(tok.config().bos_token, Some(3));
        assert_eq!(tok.config().eos_token, Some(4));
    }

    #[test]
    fn test_load_json_no_added_tokens() {
        let json = r#"{
            "model": {
                "type": "BPE",
                "vocab": {"x": 0, "y": 1},
                "merges": []
            }
        }"#;

        let tok = BpeTokenizer::from_json(json).unwrap();
        assert_eq!(tok.vocab_size(), 2);
        assert_eq!(tok.config().bos_token, None);
        assert_eq!(tok.config().eos_token, None);
    }

    #[test]
    fn test_load_json_unsupported_type() {
        let json = r#"{
            "model": {
                "type": "WordPiece",
                "vocab": {},
                "merges": []
            }
        }"#;

        assert!(BpeTokenizer::from_json(json).is_err());
    }

    #[test]
    fn test_load_json_missing_model() {
        let json = r#"{"not_model": {}}"#;
        assert!(BpeTokenizer::from_json(json).is_err());
    }
}
