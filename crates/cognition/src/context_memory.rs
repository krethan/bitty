use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::{bundle, HyperVector};

/// Streaming hypervector memory with a dense-window cutover.
///
/// Recent tokens are kept as individual dense keys; once the window fills, the
/// oldest `chunk_items` tokens are evicted together into a single bundled
/// record (majority-vote `bundle`). Retrieval probes the records by similarity
/// to a query bundle.
///
/// The bundle size is the Phase 3 spike's cutover point `n*`: at `dims = 1024`
/// a 16-item record recalls at ~1.0 with an 8-token window probe, compressing
/// evicted history ~16x versus keeping it dense.
#[derive(Debug, Clone)]
pub struct ContextMemoryConfig {
    /// Hypervector dimensionality in bits.
    pub dims: usize,
    /// Token vocabulary size; the codebook maps `0..vocab_size` to keys.
    pub vocab_size: usize,
    /// Items bundled per evicted record (the dense-window cutover, `n*`).
    pub chunk_items: usize,
    /// Max recent tokens kept as individual dense keys.
    pub window: usize,
    /// Minimum record similarity for a probe to return a result.
    pub min_similarity: f32,
    /// Seed for the deterministic token codebook.
    pub seed: u64,
}

impl Default for ContextMemoryConfig {
    fn default() -> Self {
        Self {
            dims: 1024,
            vocab_size: 96,
            chunk_items: 16,
            window: 16,
            min_similarity: 0.5,
            seed: 0x5EED_C0DE,
        }
    }
}

/// An evicted chunk: the bundled key plus the exact tokens it compressed.
#[derive(Debug, Clone)]
pub struct ChunkRecord {
    pub key: HyperVector,
    pub tokens: Vec<u32>,
    /// Stream position of the first token in `tokens`.
    pub start: usize,
}

#[derive(Debug, Clone)]
pub struct ContextMemory {
    config: ContextMemoryConfig,
    codebook: Vec<HyperVector>,
    window_keys: Vec<HyperVector>,
    window_tokens: Vec<u32>,
    records: Vec<ChunkRecord>,
    position: usize,
}

impl ContextMemory {
    pub fn new(config: ContextMemoryConfig) -> Self {
        let mut rng = StdRng::seed_from_u64(config.seed);
        let codebook = (0..config.vocab_size)
            .map(|_| HyperVector::random_with(config.dims, &mut rng))
            .collect();
        Self {
            config,
            codebook,
            window_keys: Vec::new(),
            window_tokens: Vec::new(),
            records: Vec::new(),
            position: 0,
        }
    }

    pub fn config(&self) -> &ContextMemoryConfig {
        &self.config
    }

    /// The deterministic hypervector for a token (shared codebook).
    pub fn key(&self, token: u32) -> &HyperVector {
        &self.codebook[token as usize]
    }

    /// Append a token to the stream, evicting the oldest window chunk into a
    /// bundled record once the dense window is full.
    pub fn push(&mut self, token: u32) {
        let key = self.key(token).clone();
        self.window_keys.push(key);
        self.window_tokens.push(token);
        self.position += 1;
        self.evict();
    }

    /// Best-matching record above `min_similarity`, or `None`.
    pub fn probe(&self, query: &HyperVector) -> Option<(f32, &ChunkRecord)> {
        self.records
            .iter()
            .map(|r| (query.similarity(&r.key), r))
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .filter(|(sim, _)| *sim >= self.config.min_similarity)
    }

    /// Top-k best-matching records above `min_similarity`.
    pub fn probe_top_k(&self, query: &HyperVector, k: usize) -> Vec<(f32, &ChunkRecord)> {
        let mut items: Vec<_> = self
            .records
            .iter()
            .map(|r| (query.similarity(&r.key), r))
            .filter(|(sim, _)| *sim >= self.config.min_similarity)
            .collect();
        items.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        items.truncate(k);
        items
    }

    /// Bundle of the current dense-window keys (empty window -> zero vector).
    pub fn window_bundle(&self) -> HyperVector {
        if self.window_keys.is_empty() {
            return HyperVector::new(self.config.dims);
        }
        let refs: Vec<&HyperVector> = self.window_keys.iter().collect();
        bundle(&refs)
    }

    /// Bytes of key storage (window dense + records bundled).
    pub fn memory_bytes(&self) -> usize {
        let per_key = self.config.dims.div_ceil(8);
        self.window_keys.len() * per_key + self.records.iter().map(|_| per_key).sum::<usize>()
    }

    /// Total tokens held in records (evicted history).
    pub fn record_items(&self) -> usize {
        self.records.iter().map(|r| r.tokens.len()).sum()
    }

    /// Total tokens held dense in the window.
    pub fn window_len(&self) -> usize {
        self.window_keys.len()
    }

    pub fn n_records(&self) -> usize {
        self.records.len()
    }

    pub fn records(&self) -> &[ChunkRecord] {
        &self.records
    }

    fn evict(&mut self) {
        while self.window_keys.len() > self.config.window {
            let take = self.config.chunk_items.min(self.window_keys.len());
            let tokens: Vec<u32> = self.window_tokens.drain(..take).collect();
            let keys: Vec<HyperVector> = self.window_keys.drain(..take).collect();
            let start = self.position - self.window_keys.len() - take;
            let refs: Vec<&HyperVector> = keys.iter().collect();
            self.records.push(ChunkRecord {
                key: bundle(&refs),
                tokens,
                start,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> ContextMemoryConfig {
        ContextMemoryConfig {
            dims: 1024,
            vocab_size: 128,
            chunk_items: 16,
            window: 16,
            min_similarity: 0.5,
            seed: 0x1234,
        }
    }

    #[test]
    fn test_eviction_cutover() {
        let mut mem = ContextMemory::new(test_config());
        for t in 0..272u32 {
            mem.push(t % 96);
        }
        // 256 of 272 tokens evicted into 16-token records; last 16 stay dense.
        assert_eq!(mem.n_records(), 16);
        assert_eq!(mem.record_items(), 256);
        assert_eq!(mem.window_len(), 16);
        assert_eq!(mem.records()[0].start, 0);
        assert_eq!(mem.records()[15].start, 240);
        assert_eq!(mem.records()[0].tokens.len(), 16);
        assert_eq!(mem.records()[0].tokens[0], 0);
    }

    #[test]
    fn test_probe_recovers_evicted_key() {
        let mut mem = ContextMemory::new(test_config());
        // history: record i = [key_i (8 tokens), value_i, 7 tail tokens],
        // each record 16-aligned so a key never straddles two records.
        let mut keys: Vec<Vec<u32>> = Vec::new();
        let mut values: Vec<u32> = Vec::new();
        for i in 0..16u32 {
            let k: Vec<u32> = (0..8).map(|j| i * 8 + j).collect();
            keys.push(k);
            values.push(64 + (i * 7) % 32);
            for t in &keys[i as usize] {
                mem.push(*t);
            }
            mem.push(values[i as usize]);
            for j in 0..7 {
                mem.push(64 + (i * 5 + j) % 32);
            }
        }
        // trailing filler so every pair is fully evicted
        for t in 0..16u32 {
            mem.push(t % 96);
        }

        assert_eq!(mem.n_records(), 16);
        assert_eq!(mem.window_len(), 16);
        let mut hits = 0usize;
        for i in 0..16usize {
            let refs: Vec<&HyperVector> = keys[i].iter().map(|t| mem.key(*t)).collect();
            let query = bundle(&refs);
            let (sim, rec) = mem.probe(&query).expect("probe must return a record");
            assert!(sim >= 0.5, "probe similarity must clear the threshold");
            if rec.start == i * 16 && rec.tokens[8] == values[i] {
                hits += 1;
            }
        }
        assert!(hits >= 15, "expected ~all keys recalled, got {hits}/16");
    }

    #[test]
    fn test_footprint_accounting() {
        let mut mem = ContextMemory::new(test_config());
        for t in 0..272u32 {
            mem.push(t % 96);
        }
        let per_key = test_config().dims.div_ceil(8);
        assert_eq!(
            mem.memory_bytes(),
            (mem.window_len() + mem.n_records()) * per_key
        );
        assert!(mem.memory_bytes() < 272 * per_key);
    }

    #[test]
    fn test_codebook_deterministic() {
        let a = ContextMemory::new(test_config());
        let b = ContextMemory::new(test_config());
        for t in 0..96u32 {
            assert_eq!(
                a.key(t).as_slice(),
                b.key(t).as_slice(),
                "codebook must be deterministic for token {t}"
            );
        }
    }

    #[test]
    fn test_window_bundle_empty() {
        let mem = ContextMemory::new(test_config());
        assert_eq!(mem.window_bundle().dims(), mem.config().dims);
    }
}
