use crate::HyperVector;

/// Sparse associative memory: content-addressable storage
/// that retrieves items by nearest-neighbor similarity.
///
/// Items are stored as key-value pairs where the key is a HyperVector
/// and retrieval is by similarity to a query vector.
///
/// This is the core primitive for replacing dense attention with
/// sparse content-based retrieval.
#[derive(Debug, Clone)]
pub struct SparseAssociativeMemory<T: Clone> {
    keys: Vec<HyperVector>,
    values: Vec<T>,
}

impl<T: Clone> SparseAssociativeMemory<T> {
    pub fn new() -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            keys: Vec::with_capacity(cap),
            values: Vec::with_capacity(cap),
        }
    }

    pub fn store(&mut self, key: HyperVector, value: T) {
        self.keys.push(key);
        self.values.push(value);
    }

    pub fn recall(&self, query: &HyperVector) -> Option<(f32, &T)> {
        self.keys
            .iter()
            .zip(self.values.iter())
            .map(|(k, v)| (query.similarity(k), v))
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
    }

    pub fn recall_top_k(&self, query: &HyperVector, k: usize) -> Vec<(f32, &T)> {
        if k == 0 {
            return Vec::new();
        }
        let mut items: Vec<_> = self
            .keys
            .iter()
            .zip(self.values.iter())
            .map(|(k, v)| (query.similarity(k), v))
            .collect();
        items.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        items.truncate(k);
        items
    }

    pub fn recall_above(&self, query: &HyperVector, threshold: f32) -> Vec<(f32, &T)> {
        self.keys
            .iter()
            .zip(self.values.iter())
            .map(|(k, v)| (query.similarity(k), v))
            .filter(|(sim, _)| *sim >= threshold)
            .collect()
    }

    pub fn recall_best_n(&self, query: &HyperVector, max_results: usize, min_similarity: f32) -> Vec<(f32, &T)> {
        let mut items: Vec<_> = self
            .keys
            .iter()
            .zip(self.values.iter())
            .map(|(k, v)| (query.similarity(k), v))
            .filter(|(sim, _)| *sim >= min_similarity)
            .collect();
        items.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        items.truncate(max_results);
        items
    }

    pub fn merge(&mut self, other: Self) {
        self.keys.extend(other.keys);
        self.values.extend(other.values);
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    pub fn clear(&mut self) {
        self.keys.clear();
        self.values.clear();
    }

    pub fn keys(&self) -> &[HyperVector] {
        &self.keys
    }

    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// Retrieve the key at a given index
    pub fn get_key(&self, idx: usize) -> Option<&HyperVector> {
        self.keys.get(idx)
    }

    /// Retrieve the value at a given index
    pub fn get_value(&self, idx: usize) -> Option<&T> {
        self.values.get(idx)
    }
}

impl<T: Clone> Default for SparseAssociativeMemory<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> FromIterator<(HyperVector, T)> for SparseAssociativeMemory<T> {
    fn from_iter<I: IntoIterator<Item = (HyperVector, T)>>(iter: I) -> Self {
        let (keys, values) = iter.into_iter().unzip();
        Self { keys, values }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_and_recall() {
        let mut mem = SparseAssociativeMemory::new();
        let k1 = HyperVector::random(256);
        let k2 = HyperVector::random(256);

        mem.store(k1.clone(), "hello");
        mem.store(k2.clone(), "world");

        let (sim, val) = mem.recall(&k1).unwrap();
        assert!((sim - 1.0).abs() < 1e-6);
        assert_eq!(*val, "hello");
    }

    #[test]
    fn test_recall_top_k() {
        let mut mem = SparseAssociativeMemory::new();
        let query = HyperVector::random(256);

        for i in 0..10 {
            let mut k = query.clone();
            k.flip_bit(i);
            mem.store(k, i);
        }

        let top3 = mem.recall_top_k(&query, 3);
        assert_eq!(top3.len(), 3);
        for (sim, _) in &top3 {
            assert!(*sim > 0.9);
        }
    }

    #[test]
    fn test_recall_above_threshold() {
        let mut mem = SparseAssociativeMemory::new();
        let query = HyperVector::random(128);

        let close = query.clone();
        let mut far = query.clone();
            for i in 0..65 {
                far.flip_bit(i);
            }

        mem.store(close.clone(), "close");
        mem.store(far.clone(), "far");

        let above = mem.recall_above(&query, 0.5);
        assert_eq!(above.len(), 1);
        assert_eq!(*above[0].1, "close");
    }

    #[test]
    fn test_empty_recall() {
        let mem: SparseAssociativeMemory<i32> = SparseAssociativeMemory::new();
        let query = HyperVector::random(64);
        assert!(mem.recall(&query).is_none());
    }

    #[test]
    fn test_clear() {
        let mut mem = SparseAssociativeMemory::new();
        mem.store(HyperVector::random(64), 1);
        assert!(!mem.is_empty());
        mem.clear();
        assert!(mem.is_empty());
    }

    #[test]
    fn test_from_iter() {
        let items = vec![
            (HyperVector::random(64), "a"),
            (HyperVector::random(64), "b"),
        ];
        let mem: SparseAssociativeMemory<&str> = items.into_iter().collect();
        assert_eq!(mem.len(), 2);
    }
}
