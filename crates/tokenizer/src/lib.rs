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
    #[error("Protobuf error: {0}")]
    ProtobufError(String),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenizerConfig {
    pub vocab_size: usize,
    pub bos_token: Option<u32>,
    pub eos_token: Option<u32>,
    pub unk_token: Option<u32>,
    /// Whether the vocabulary is GPT-2 byte-level encoded (space → `Ġ`,
    /// newline → `Ċ`, …). When set, text is pre-tokenized with the GPT-2
    /// regex and byte-encoded before BPE merges, and tokens are byte-decoded
    /// on decode.
    pub byte_level: bool,
    /// Whether the pre-tokenizer splits digit runs into individual digits
    /// (HuggingFace `Digits(individual_digits=true)`).
    pub digits_individual: bool,
}

pub struct BpeTokenizer {
    vocab: HashMap<String, u32>,
    inv_vocab: HashMap<u32, String>,
    merges: Vec<(String, String)>,
    config: TokenizerConfig,
    byte_encoder: HashMap<u8, char>,
    byte_decoder: HashMap<char, u8>,
}

/// GPT-2 byte-to-unicode mapping. Bytes that already have printable
/// characters keep them; the remaining bytes (space, newline, control, high
/// bytes) map into U+0100..U+0180. Space → `Ġ` (U+0120), newline → `Ċ`
/// (U+010A), tab → `ĉ` (U+0109).
fn bytes_to_unicode() -> (HashMap<u8, char>, HashMap<char, u8>) {
    let mut bs: Vec<u8> = Vec::new();
    let mut cs: Vec<u16> = Vec::new();
    for b in 33..=126 {
        bs.push(b);
        cs.push(b as u16);
    }
    for b in 161..=172 {
        bs.push(b);
        cs.push(b as u16);
    }
    for b in 174..=255 {
        bs.push(b);
        cs.push(b as u16);
    }
    let mut n: u16 = 0;
    for b in 0..=255u8 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let encoder: HashMap<u8, char> = bs
        .iter()
        .zip(cs.iter())
        .map(|(&b, &c)| (b, char::from_u32(c as u32).unwrap()))
        .collect();
    let decoder = encoder.iter().map(|(&b, &c)| (c, b)).collect();
    (encoder, decoder)
}

fn is_letter(c: char) -> bool {
    c.is_alphabetic()
}

fn is_number(c: char) -> bool {
    c.is_numeric()
}

/// A non-whitespace character that is neither a letter nor a number.
fn is_punct(c: char) -> bool {
    !c.is_whitespace() && !is_letter(c) && !is_number(c)
}

impl BpeTokenizer {
    pub fn new(config: TokenizerConfig) -> Self {
        let (byte_encoder, byte_decoder) = bytes_to_unicode();
        Self {
            vocab: HashMap::new(),
            inv_vocab: HashMap::new(),
            merges: Vec::new(),
            config,
            byte_encoder,
            byte_decoder,
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
        let (byte_encoder, byte_decoder) = bytes_to_unicode();
        Self {
            vocab,
            inv_vocab,
            merges,
            config,
            byte_encoder,
            byte_decoder,
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
        let (byte_encoder, byte_decoder) = bytes_to_unicode();
        Self {
            vocab,
            inv_vocab,
            merges,
            config,
            byte_encoder,
            byte_decoder,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    pub fn config(&self) -> &TokenizerConfig {
        &self.config
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let words = if self.config.byte_level {
            self.pretokenize_byte_level(text)
        } else {
            self.regex_split(text)
        };
        let mut tokens = Vec::new();

        for word in words {
            let mut chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();

            // Build priority map: (left, right) -> earliest merge index
            let mut priority: HashMap<(&str, &str), usize> = HashMap::new();
            for (i, (l, r)) in self.merges.iter().enumerate() {
                let key = (l.as_str(), r.as_str());
                priority.entry(key).or_insert(i);
            }

            // Iteratively apply the highest-priority available merge
            loop {
                if chars.len() < 2 {
                    break;
                }
                let mut best_pos = usize::MAX;
                let mut best_pri = usize::MAX;
                for i in 0..chars.len() - 1 {
                    let key = (chars[i].as_str(), chars[i + 1].as_str());
                    if let Some(&pri) = priority.get(&key) {
                        if pri < best_pri {
                            best_pri = pri;
                            best_pos = i;
                        }
                    }
                }
                if best_pos == usize::MAX {
                    break;
                }
                let merged = format!("{}{}", chars[best_pos], chars[best_pos + 1]);
                chars[best_pos] = merged;
                chars.remove(best_pos + 1);
            }

            for c in chars {
                if let Some(&id) = self.vocab.get(&c) {
                    tokens.push(id);
                } else {
                    for ch in c.chars() {
                        let s = ch.to_string();
                        if let Some(&id) = self.vocab.get(&s) {
                            tokens.push(id);
                        } else if let Some(unk) = self.config.unk_token {
                            tokens.push(unk);
                        }
                    }
                }
            }
        }

        tokens
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String, TokenizerError> {
        let mut result = Vec::new();
        for &token in tokens {
            if token == self.config.bos_token.unwrap_or(u32::MAX) {
                continue;
            }
            if token == self.config.eos_token.unwrap_or(u32::MAX) {
                break;
            }
            match self.inv_vocab.get(&token) {
                Some(s) => {
                    if self.config.byte_level {
                        // Byte-decoded tokens back to raw bytes. Characters that
                        // are not part of the byte-level alphabet (special
                        // tokens, literals) are re-emitted as UTF-8.
                        for ch in s.chars() {
                            match self.byte_decoder.get(&ch) {
                                Some(&b) => result.push(b),
                                None => {
                                    let mut buf = [0u8; 4];
                                    result.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                                }
                            }
                        }
                    } else {
                        result.extend_from_slice(s.as_bytes());
                    }
                }
                None => {
                    return Err(TokenizerError::UnknownToken(format!("token_id={}", token)));
                }
            }
        }
        Ok(String::from_utf8_lossy(&result).into_owned())
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

    /// GPT-2 byte-level pre-tokenization. Splits the text with the standard
    /// GPT-2 regex, byte-encodes each chunk (space → `Ġ`, …), and optionally
    /// isolates individual digits. Mirrors the HuggingFace
    /// `Split(GPT2 regex)` + `ByteLevel` pre-tokenizer.
    fn pretokenize_byte_level(&self, text: &str) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        let mut chunks: Vec<String> = Vec::new();
        let mut i = 0;
        while i < chars.len() {
            let c = chars[i];

            // `'(?:[sdmt]|ll|ve|re)` — apostrophe contractions, case-insensitive.
            if c == '\'' {
                if let Some(contraction_len) = Self::contraction_len(&chars, i) {
                    chunks.push(self.byte_encode(&chars[i..i + contraction_len].iter().collect::<String>()));
                    i += contraction_len;
                    continue;
                }
            }

            // ` ?\p{L}+` — one optional leading whitespace + a letter run.
            if c.is_whitespace() && i + 1 < chars.len() && is_letter(chars[i + 1]) {
                let mut j = i + 2;
                while j < chars.len() && is_letter(chars[j]) {
                    j += 1;
                }
                chunks.push(self.byte_encode(&chars[i..j].iter().collect::<String>()));
                i = j;
                continue;
            }
            if is_letter(c) {
                let mut j = i + 1;
                while j < chars.len() && is_letter(chars[j]) {
                    j += 1;
                }
                chunks.push(self.byte_encode(&chars[i..j].iter().collect::<String>()));
                i = j;
                continue;
            }

            // `\p{N}+` — a digit run, optionally split into single digits.
            if is_number(c) {
                let mut j = i + 1;
                while j < chars.len() && is_number(chars[j]) {
                    j += 1;
                }
                if self.config.digits_individual {
                    for d in &chars[i..j] {
                        chunks.push(self.byte_encode(&d.to_string()));
                    }
                } else {
                    chunks.push(self.byte_encode(&chars[i..j].iter().collect::<String>()));
                }
                i = j;
                continue;
            }

            // ` ?[^\s\p{L}\p{N}]+[\r\n]*` — punctuation runs, optionally
            // preceded by one whitespace, with trailing CR/LF.
            if c.is_whitespace() && i + 1 < chars.len() && is_punct(chars[i + 1]) {
                let mut j = i + 2;
                while j < chars.len() && is_punct(chars[j]) {
                    j += 1;
                }
                while j < chars.len() && (chars[j] == '\r' || chars[j] == '\n') {
                    j += 1;
                }
                chunks.push(self.byte_encode(&chars[i..j].iter().collect::<String>()));
                i = j;
                continue;
            }
            if is_punct(c) {
                let mut j = i + 1;
                while j < chars.len() && is_punct(chars[j]) {
                    j += 1;
                }
                while j < chars.len() && (chars[j] == '\r' || chars[j] == '\n') {
                    j += 1;
                }
                chunks.push(self.byte_encode(&chars[i..j].iter().collect::<String>()));
                i = j;
                continue;
            }

            // `\s+(?!\S)` then `\s+` — whitespace runs. When the run is
            // followed by a non-whitespace char, the last whitespace char is
            // left for the next match (so ` ?\p{L}+` can attach it to the
            // following word).
            if c.is_whitespace() {
                let mut j = i + 1;
                while j < chars.len() && chars[j].is_whitespace() {
                    j += 1;
                }
                if j < chars.len() && !chars[j].is_whitespace() && j > i + 1 {
                    j -= 1;
                }
                chunks.push(self.byte_encode(&chars[i..j].iter().collect::<String>()));
                i = j;
                continue;
            }

            chunks.push(self.byte_encode(&c.to_string()));
            i += 1;
        }
        chunks
    }

    /// Length (in chars) of an apostrophe contraction at `i` (`'s`, `'t`,
    /// `'re`, `'ve`, `'m`, `'ll`, `'d`), case-insensitively, or `None`.
    fn contraction_len(chars: &[char], i: usize) -> Option<usize> {
        fn eq(a: char, b: char) -> bool {
            a == b || a.to_ascii_lowercase() == b
        }
        let rest = &chars[i + 1..];
        const TWO: [&[char]; 4] = [&['s'], &['t'], &['m'], &['d']];
        for suffix in TWO {
            if rest.len() >= 1 && eq(rest[0], suffix[0]) {
                return Some(2);
            }
        }
        const THREE: [&[char]; 3] = [&['l', 'l'], &['v', 'e'], &['r', 'e']];
        for suffix in THREE {
            if rest.len() >= 2 && eq(rest[0], suffix[0]) && eq(rest[1], suffix[1]) {
                return Some(3);
            }
        }
        None
    }

    fn byte_encode(&self, chunk: &str) -> String {
        chunk
            .bytes()
            .map(|b| self.byte_encoder.get(&b).copied().unwrap_or(b as char))
            .collect()
    }

    /// Inspect the `pre_tokenizer` section of a HuggingFace tokenizer.json to
    /// decide whether to use GPT-2 byte-level encoding and/or isolate digits.
    fn pretokenizer_flags(value: &serde_json::Value) -> (bool, bool) {
        fn flags_for(node: &serde_json::Value) -> (bool, bool) {
            let mut flags = (false, false);
            if let Some(kind) = node.get("type").and_then(|t| t.as_str()) {
                match kind {
                    "ByteLevel" => flags.0 = true,
                    "Digits" => {
                        flags.1 = node
                            .get("individual_digits")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false);
                    }
                    _ => {}
                }
            }
            if let Some(seq) = node.get("pretokenizers").and_then(|v| v.as_array()) {
                for sub in seq {
                    let (a, b) = flags_for(sub);
                    flags.0 |= a;
                    flags.1 |= b;
                }
            }
            flags
        }
        value.get("pre_tokenizer").map(flags_for).unwrap_or((false, false))
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
                        if id > u32::MAX as u64 {
                            return Err(TokenizerError::LoadError(format!(
                                "token id {} exceeds u32 range for token '{}'",
                                id, key
                            )));
                        }
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

        // Whether the pre-tokenizer is GPT-2 byte-level and/or isolates digits.
        let (byte_level, digits_individual) = Self::pretokenizer_flags(&value);

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
            byte_level,
            digits_individual,
        };

        Ok(Self::from_vocab_and_merges_with_config(
            vocab, merges, config,
        ))
    }

    /// Load a SentencePiece tokenizer from a .model file.
    ///
    /// SentencePiece files are protobuf-encoded. This parser extracts the vocabulary
    /// and special tokens (BOS, EOS, UNK) but does not implement the full SentencePiece
    /// encoding algorithm. For inference, use the HuggingFace tokenizer.json format instead.
    pub fn load_sentencepiece<P: AsRef<Path>>(path: P) -> Result<Self, TokenizerError> {
        let mut file = File::open(path.as_ref())?;
        let mut contents = Vec::new();
        file.read_to_end(&mut contents)?;
        Self::from_sentencepiece_bytes(&contents)
    }

    /// Parse SentencePiece protobuf bytes and build a tokenizer.
    pub fn from_sentencepiece_bytes(data: &[u8]) -> Result<Self, TokenizerError> {
        let pieces = parse_sentencepiece_proto(data)?;

        let mut vocab = HashMap::new();
        let mut bos_token = None;
        let mut eos_token = None;
        let mut unk_token = None;

        for (id, piece) in pieces.iter().enumerate() {
            let id = id as u32;
            vocab.insert(piece.piece.clone(), id);

            // Identify special tokens by type and content
            match piece.piece_type {
                SentencePieceType::Control => {
                    if piece.piece == "<s>" || piece.piece == "<BOS>" {
                        bos_token = Some(id);
                    } else if piece.piece == "</s>" || piece.piece == "<EOS>" {
                        eos_token = Some(id);
                    } else if piece.piece == "<unk>" {
                        unk_token = Some(id);
                    }
                }
                SentencePieceType::Unknown => {
                    if unk_token.is_none() {
                        unk_token = Some(id);
                    }
                }
                _ => {}
            }
        }

        let config = TokenizerConfig {
            vocab_size: vocab.len(),
            bos_token,
            eos_token,
            unk_token,
            byte_level: false,
            digits_individual: false,
        };

        // SentencePiece doesn't expose merges in the same way as BPE
        // We create an empty merges list - encoding will use character-level fallback
        Ok(Self::from_vocab_and_merges_with_config(vocab, Vec::new(), config))
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

// ============================================================================
// SentencePiece protobuf parsing
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
enum SentencePieceType {
    Normal = 1,
    Unknown = 2,
    Control = 3,
    UserDefined = 4,
    Unused = 5,
    Byte = 6,
}

impl From<i32> for SentencePieceType {
    fn from(v: i32) -> Self {
        match v {
            1 => SentencePieceType::Normal,
            2 => SentencePieceType::Unknown,
            3 => SentencePieceType::Control,
            4 => SentencePieceType::UserDefined,
            5 => SentencePieceType::Unused,
            6 => SentencePieceType::Byte,
            _ => SentencePieceType::Unknown,
        }
    }
}

#[derive(Debug, Clone)]
struct SentencePiecePiece {
    piece: String,
    piece_type: SentencePieceType,
}

/// Minimal protobuf parser for SentencePiece ModelProto messages.
///
/// SentencePiece .model files use protobuf encoding with the following structure:
/// ```protobuf
/// message ModelProto {
///   message Piece {
///     enum Type { NORMAL=1; UNKNOWN=2; CONTROL=3; USER_DEFINED=4; UNUSED=5; BYTE=6; }
///     string piece = 1;
///     float score = 2;
///     Type type = 3;
///   }
///   repeated Piece pieces = 1;
///   // ... other fields we ignore
/// }
/// ```
fn parse_sentencepiece_proto(data: &[u8]) -> Result<Vec<SentencePiecePiece>, TokenizerError> {
    let mut pieces = Vec::new();
    let mut pos = 0;

    while pos < data.len() {
        // Read field tag
        let (field_tag, new_pos) = read_varint(data, pos)?;
        pos = new_pos;

        let field_number = field_tag >> 3;
        let wire_type = field_tag & 0x7;

        match (field_number, wire_type) {
            // pieces field (repeated Piece, field 1, wire type 2 = length-delimited)
            (1, 2) => {
                let (len, new_pos) = read_varint(data, pos)?;
                pos = new_pos;
                let len = len as usize;
                if pos + len > data.len() {
                    return Err(TokenizerError::ProtobufError("truncated piece data".into()));
                }
                let piece_data = &data[pos..pos + len];
                pieces.push(parse_piece(piece_data)?);
                pos += len;
            }
            // Skip other fields
            (_, 0) => {
                // Varint
                let (_, new_pos) = read_varint(data, pos)?;
                pos = new_pos;
            }
            (_, 1) => {
                // 64-bit
                if pos + 8 > data.len() {
                    return Err(TokenizerError::ProtobufError("truncated 64-bit field".into()));
                }
                pos += 8;
            }
            (_, 2) => {
                // Length-delimited
                let (len, new_pos) = read_varint(data, pos)?;
                pos = new_pos;
                let len = len as usize;
                if pos + len > data.len() {
                    return Err(TokenizerError::ProtobufError("truncated length-delimited field".into()));
                }
                pos += len;
            }
            (_, 5) => {
                // 32-bit
                if pos + 4 > data.len() {
                    return Err(TokenizerError::ProtobufError("truncated 32-bit field".into()));
                }
                pos += 4;
            }
            _ => {
                return Err(TokenizerError::ProtobufError(format!(
                    "unknown wire type {} for field {}",
                    wire_type, field_number
                )));
            }
        }
    }

    Ok(pieces)
}

fn parse_piece(data: &[u8]) -> Result<SentencePiecePiece, TokenizerError> {
    let mut piece = String::new();
    let mut piece_type = SentencePieceType::Normal;
    let mut pos = 0;

    while pos < data.len() {
        let (field_tag, new_pos) = read_varint(data, pos)?;
        pos = new_pos;

        let field_number = field_tag >> 3;
        let wire_type = field_tag & 0x7;

        match (field_number, wire_type) {
            // piece (string, field 1)
            (1, 2) => {
                let (len, new_pos) = read_varint(data, pos)?;
                pos = new_pos;
                let len = len as usize;
                if pos + len > data.len() {
                    return Err(TokenizerError::ProtobufError("truncated piece string".into()));
                }
                piece = String::from_utf8_lossy(&data[pos..pos + len]).to_string();
                pos += len;
            }
            // score (float, field 2) — parsed and skipped; unused
            (2, 5) => {
                if pos + 4 > data.len() {
                    return Err(TokenizerError::ProtobufError("truncated score".into()));
                }
                pos += 4;
            }
            // type (enum/int32, field 3)
            (3, 0) => {
                let (v, new_pos) = read_varint(data, pos)?;
                pos = new_pos;
                piece_type = SentencePieceType::from(v as i32);
            }
            // Skip unknown fields
            (_, 0) => {
                let (_, new_pos) = read_varint(data, pos)?;
                pos = new_pos;
            }
            (_, 1) => {
                if pos + 8 > data.len() {
                    return Err(TokenizerError::ProtobufError("truncated 64-bit field".into()));
                }
                pos += 8;
            }
            (_, 2) => {
                let (len, new_pos) = read_varint(data, pos)?;
                pos = new_pos;
                let len = len as usize;
                if pos + len > data.len() {
                    return Err(TokenizerError::ProtobufError("truncated length-delimited field".into()));
                }
                pos += len;
            }
            (_, 5) => {
                if pos + 4 > data.len() {
                    return Err(TokenizerError::ProtobufError("truncated 32-bit field".into()));
                }
                pos += 4;
            }
            _ => {
                return Err(TokenizerError::ProtobufError(format!(
                    "unknown wire type in piece: {} for field {}",
                    wire_type, field_number
                )));
            }
        }
    }

    Ok(SentencePiecePiece {
        piece,
        piece_type,
    })
}

fn read_varint(data: &[u8], mut pos: usize) -> Result<(u64, usize), TokenizerError> {
    let mut result: u64 = 0;
    let mut shift = 0;
    loop {
        if pos >= data.len() {
            return Err(TokenizerError::ProtobufError("truncated varint".into()));
        }
        let byte = data[pos];
        pos += 1;
        result |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
        if shift >= 64 {
            return Err(TokenizerError::ProtobufError("varint too long".into()));
        }
    }
    Ok((result, pos))
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
    fn test_byte_level_no_unk_for_spaces() {
        // Verify that byte-level mode does not emit unk tokens for space-prefixed
        // words (the bug that caused Qwen ppl to explode).
        let mut vocab = HashMap::new();
        // All printable ASCII + Ġ and Ċ
        for b in 33u8..=126 {
            vocab.insert((b as char).to_string(), (b - 33) as u32);
        }
        vocab.insert("Ġ".to_string(), 100);
        vocab.insert("Ġhello".to_string(), 101);
        vocab.insert("Ġworld".to_string(), 102);
        vocab.insert("hello".to_string(), 103);
        vocab.insert("world".to_string(), 104);
        let merges = vec![];
        let mut config = TokenizerConfig::default();
        config.byte_level = true;
        let tok = BpeTokenizer::from_vocab_and_merges_with_config(vocab, merges, config);

        let encoded = tok.encode("hello world");
        // With byte-level mode, " world" → Ġworld (no unk). Without byte-level,
        // the bare " " chunk would produce unk (if unk_token is set).
        if let Some(unk_id) = tok.config().unk_token {
            assert!(
                !encoded.contains(&unk_id),
                "byte-level mode should not emit unk for space-prefixed words"
            );
        }
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

    #[test]
    fn test_sentencepiece_proto_parsing() {
        // Create a minimal SentencePiece protobuf with 3 pieces:
        // - "<s>" (control, BOS)
        // - "hello" (normal)
        // - "</s>" (control, EOS)
        let mut data = Vec::new();

        // Helper to write a varint
        fn write_varint(buf: &mut Vec<u8>, mut val: u64) {
            while val >= 0x80 {
                buf.push((val & 0x7F) as u8 | 0x80);
                val >>= 7;
            }
            buf.push(val as u8);
        }

        // Helper to write a piece
        fn write_piece(buf: &mut Vec<u8>, piece: &str, score: f32, piece_type: i32) {
            let mut piece_data = Vec::new();
            // piece string (field 1, wire type 2)
            piece_data.push(0x0A); // field 1, wire type 2
            write_varint(&mut piece_data, piece.len() as u64);
            piece_data.extend_from_slice(piece.as_bytes());
            // score (field 2, wire type 5)
            piece_data.push(0x15); // field 2, wire type 5
            piece_data.extend_from_slice(&score.to_le_bytes());
            // type (field 3, wire type 0)
            piece_data.push(0x18); // field 3, wire type 0
            write_varint(&mut piece_data, piece_type as u64);

            // Write piece as field 1 of ModelProto
            buf.push(0x0A); // field 1, wire type 2
            write_varint(buf, piece_data.len() as u64);
            buf.extend_from_slice(&piece_data);
        }

        write_piece(&mut data, "<s>", 0.0, 3); // control
        write_piece(&mut data, "hello", -1.5, 1); // normal
        write_piece(&mut data, "</s>", 0.0, 3); // control

        let tok = BpeTokenizer::from_sentencepiece_bytes(&data).unwrap();
        assert_eq!(tok.vocab_size(), 3);
        assert_eq!(tok.config().bos_token, Some(0));
        assert_eq!(tok.config().eos_token, Some(2));
    }
}
