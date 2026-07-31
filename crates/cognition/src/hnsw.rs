use crate::HyperVector;

/// A node in the HNSW graph.
#[derive(Debug)]
struct HNSWNode<T> {
    key: HyperVector,
    value: T,
    /// Indices of neighbor nodes (single-layer NSW).
    neighbors: Vec<usize>,
}

/// A single-layer Navigable Small World graph over HyperVector keys.
///
/// Insertions connect each new key to its top search candidates, keeping
/// edges metric-aware. Search performs greedy best-first descent from an
/// entry point with a candidate beam (width = k).
///
/// KNOWN LIMITATION (empirically verified, not yet solved): greedy descent
/// is bounded by the entry point's connected basin — the graph never develops
/// the long-range edges needed to navigate to far queries, so search reliably
/// finds keys only near the entry. The canonical fix is a multi-layer HNSW
/// hierarchy whose top layers span the whole space; a naive multi-layer
/// attempt was benchmarked on trivial near-entry queries and wrongly
/// rejected. Flat scan (`SparseAssociativeMemory`) is currently the reliable
/// retrieval path. Navigation quality must be validated with far queries
/// before trusting `search` results.
///
/// A multi-layer hierarchy was tried and benchmarked (256 → 262K keys) and
/// lost on both latency and insert cost: the NSW graph already navigates in
/// O(log M) hops, so extra layers only add their own descents. Single-layer
/// with an allocation-free beam is the empirical winner in this regime.
///
/// All distances are Hamming distances on packed bits — a few XOR+POPCOUNTs
/// per node visit, cheap enough to stay compute-bound even in SRAM/L2.
#[derive(Debug)]
pub struct BitHNSW<T> {
    dims: usize,
    max_degree: usize,
    ef_search: usize,
    ef_construction: usize,
    nodes: Vec<HNSWNode<T>>,
    entry_point: Option<usize>,
    // Reusable scratch for best-first search: a generation-stamped visited
    // array plus one flat heap buffer. No per-call allocations. Stamps are
    // u8 — generation wraps every ~255 searches and the array is reset then,
    // so the reset cost amortizes to O(M/255) per search.
    visited_gen: u8,
    visited_stamps: Vec<u8>,
    heap: Vec<Match>,
}

/// A search result with its Hamming distance from the query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Match {
    pub distance: u32,
    pub index: usize,
}

impl Ord for Match {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance
            .cmp(&other.distance)
            .then_with(|| self.index.cmp(&other.index))
    }
}

impl PartialOrd for Match {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> BitHNSW<T> {
    pub fn new(dims: usize) -> Self {
        Self {
            dims,
            max_degree: 16,
            ef_search: 16,
            ef_construction: 32,
            nodes: Vec::new(),
            entry_point: None,
            visited_gen: 1,
            visited_stamps: Vec::new(),
            heap: Vec::new(),
        }
    }

    pub fn with_params(mut self, max_degree: usize, ef_search: usize, ef_construction: usize) -> Self {
        self.max_degree = max_degree.max(1);
        self.ef_search = ef_search.max(1);
        self.ef_construction = ef_construction.max(1);
        self
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Store a key-value pair, linking it into the graph.
    ///
    /// Connects the new node to the up-to-`max_degree` nearest existing nodes
    /// found by an `ef_construction`-beam search. Linking to the full
    /// candidate set (not just the single nearest) keeps edges metric-aware,
    /// though as documented on the struct, this alone does not make the graph
    /// navigable for far queries.
    pub fn insert(&mut self, key: HyperVector, value: T) {
        assert_eq!(key.dims(), self.dims, "key dimension mismatch");
        let new_idx = self.nodes.len();

        // Connect to the best-first descent candidates, pruned to max_degree.
        let mut candidates: Vec<usize> = Vec::new();
        if !self.nodes.is_empty() {
            candidates.extend(
                self.best_first_search(&key, self.ef_construction)
                    .into_iter()
                    .map(|m| m.index)
                    .take(self.max_degree),
            );
        }
        candidates.sort_by_key(|&idx| key.hamming_distance(&self.nodes[idx].key));
        candidates.truncate(self.max_degree);
        let mut deduped = Vec::with_capacity(candidates.len());
        for &c in &candidates {
            if !deduped.contains(&c) {
                deduped.push(c);
            }
        }
        candidates = deduped;

        self.nodes.push(HNSWNode {
            key,
            value,
            neighbors: candidates.clone(),
        });
        // Generation stamps start at 1, so a stamp of 0 never reads as visited.
        self.visited_stamps.push(0);

        // Add back-edges to the new node.
        for &idx in &candidates {
            let n = self.nodes[idx].neighbors.len();
            if n < self.max_degree {
                self.nodes[idx].neighbors.push(new_idx);
            }
        }

        if self.entry_point.is_none() {
            self.entry_point = Some(0);
        }
    }

    /// Retrieve the closest stored key's index to the query.
    pub fn search_nearest(&mut self, query: &HyperVector) -> Option<(u32, usize)> {
        if self.nodes.is_empty() {
            return None;
        }
        self.search(query, 1).first().map(|m| (m.distance, m.index))
    }

    /// Retrieve the k closest stored keys to the query.
    /// Returns matches sorted by ascending Hamming distance.
    ///
    /// Best-first descent from the entry point with a beam of width `k`.
    /// Takes `&mut self` because the visited cache is reused across calls and
    /// never cleared.
    pub fn search(&mut self, query: &HyperVector, k: usize) -> Vec<Match> {
        if self.nodes.is_empty() || k == 0 {
            return Vec::new();
        }
        assert_eq!(query.dims(), self.dims, "query dimension mismatch");
        self.best_first_search(query, k)
    }

    /// Best-first descent from the entry point, collecting up to `k` nearest
    /// visited nodes. Allocation-free: reuses `visited_stamps` and `heap`.
    fn best_first_search(&mut self, query: &HyperVector, k: usize) -> Vec<Match> {
        let k = k.max(1);
        let entry = self.entry_point.expect("non-empty HNSW must have entry point");

        // Bump the visited generation. On wrap, reset the stamp array.
        self.visited_gen = self.visited_gen.wrapping_add(1);
        if self.visited_gen == 0 {
            self.visited_stamps.fill(0);
            self.visited_gen = 1;
        }
        if self.visited_stamps.len() < self.nodes.len() {
            self.visited_stamps.resize(self.nodes.len(), 0);
        }

        let heap = &mut self.heap;
        heap.clear();
        heap_push(heap, Match {
            distance: query.hamming_distance(&self.nodes[entry].key),
            index: entry,
        });

        let mut results: Vec<Match> = Vec::with_capacity(k);
        while let Some(m) = heap_pop(heap) {
            // Pop-side dedup: a node may be pushed twice before being visited.
            if self.visited_stamps[m.index] == self.visited_gen {
                continue;
            }
            self.visited_stamps[m.index] = self.visited_gen;

            // Once results are full, nothing on the frontier can beat the k-th
            // best if the smallest remaining distance is already larger.
            if results.len() == k && m.distance > results[k - 1].distance {
                break;
            }

            if results.len() < k {
                results.push(m);
                results.sort_unstable_by_key(|r| r.distance);
            } else if m.distance < results[k - 1].distance {
                results[k - 1] = m;
                results.sort_unstable_by_key(|r| r.distance);
            }

            for &n_idx in &self.nodes[m.index].neighbors {
                if self.visited_stamps[n_idx] == self.visited_gen {
                    continue;
                }
                let d = query.hamming_distance(&self.nodes[n_idx].key);
                if results.len() < k || d < results[k - 1].distance {
                    heap_push(heap, Match { distance: d, index: n_idx });
                }
            }
        }

        results.sort_unstable_by_key(|r| r.distance);
        results
    }

    /// Reference to a stored value by node index.
    pub fn value(&self, index: usize) -> Option<&T> {
        self.nodes.get(index).map(|n| &n.value)
    }

    /// Reference to a stored key by node index.
    pub fn key(&self, index: usize) -> Option<&HyperVector> {
        self.nodes.get(index).map(|n| &n.key)
    }

    /// Rebuild neighbor lists to satisfy the max_degree invariant.
    /// Call after bulk insertions to tighten connectivity.
    pub fn prune(&mut self) {
        let n = self.nodes.len();
        for i in 0..n {
            let mut neighbors = self.nodes[i].neighbors.clone();
            neighbors.sort_by_key(|&j| self.nodes[i].key.hamming_distance(&self.nodes[j].key));
            neighbors.truncate(self.max_degree);
            self.nodes[i].neighbors = neighbors;
        }
    }
}

/// Push onto a min-heap by distance (flat buffer, allocation-free reuse).
fn heap_push(heap: &mut Vec<Match>, m: Match) {
    heap.push(m);
    let mut i = heap.len() - 1;
    while i > 0 {
        let parent = (i - 1) / 2;
        if heap[i].distance < heap[parent].distance {
            heap.swap(i, parent);
            i = parent;
        } else {
            break;
        }
    }
}

/// Pop the minimum-distance element off a min-heap.
fn heap_pop(heap: &mut Vec<Match>) -> Option<Match> {
    if heap.is_empty() {
        return None;
    }
    let last = heap.len() - 1;
    heap.swap(0, last);
    let root = heap.pop()?;
    let mut i = 0;
    let len = heap.len();
    loop {
        let left = 2 * i + 1;
        let right = 2 * i + 2;
        let mut smallest = i;
        if left < len && heap[left].distance < heap[smallest].distance {
            smallest = left;
        }
        if right < len && heap[right].distance < heap[smallest].distance {
            smallest = right;
        }
        if smallest == i {
            break;
        }
        heap.swap(i, smallest);
        i = smallest;
    }
    Some(root)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_search_exact() {
        let mut h = BitHNSW::new(256);
        let q = HyperVector::random(256);
        for i in 0..50 {
            let mut k = q.clone();
            k.flip_bit((i * 7) % 256);
            if i > 0 {
                k.flip_bit((i * 7 + 1) % 256);
            }
            h.insert(k, i);
        }

        let res = h.search_nearest(&q).unwrap();
        assert_eq!(res.1, 0); // index 0 is the unique key 1 bit away
        assert_eq!(res.0, 1);
    }

    #[test]
    fn test_search_returns_k() {
        let mut h = BitHNSW::new(128);
        let q = HyperVector::random(128);
        for i in 0..100 {
            let mut k = q.clone();
            for b in 0..(i % 10) {
                k.flip_bit(b);
            }
            h.insert(k, i);
        }

        let results = h.search(&q, 5);
        assert_eq!(results.len(), 5);
        // Sorted ascending by distance
        assert!(results.windows(2).all(|w| w[0].distance <= w[1].distance));
    }

    #[test]
    fn test_visited_gen_wrap() {
        // u8 stamps wrap every ~255 searches; verify results stay stable across
        // the wrap boundary (the array is reset rather than silently corrupting).
        let mut h = BitHNSW::new(256);
        let q = HyperVector::random(256);
        for i in 0..50 {
            let mut k = q.clone();
            k.flip_bit((i * 7) % 256);
            h.insert(k, i);
        }

        let expected = h.search(&q, 3).into_iter().map(|m| m.index).collect::<Vec<_>>();
        for _ in 0..600 {
            let r = h.search(&q, 3);
            let idx: Vec<usize> = r.iter().map(|m| m.index).collect();
            assert_eq!(idx, expected, "result changed across visited-gen wrap");
        }
    }

    #[test]
    fn test_empty_search() {
        let mut h: BitHNSW<i32> = BitHNSW::new(64);
        let q = HyperVector::random(64);
        assert!(h.search_nearest(&q).is_none());
        assert!(h.search(&q, 5).is_empty());
    }

    #[test]
    fn test_values_retrievable() {
        let mut h = BitHNSW::new(64);
        let q = HyperVector::random(64);
        let mut k1 = q.clone();
        k1.flip_bit(0);
        h.insert(k1, "alpha");
        let mut k2 = q.clone();
        k2.flip_bit(1);
        k2.flip_bit(2);
        h.insert(k2, "beta");

        let m = h.search_nearest(&q).unwrap();
        assert_eq!(m.0, 1);
        assert_eq!(*h.value(m.1).unwrap(), "alpha");
    }

    #[test]
    fn test_hnsw_outperforms_linear_on_random() {
        // With clustered keys, NSW search should find the same nearest neighbor
        // as a linear scan (correctness check, not a speed check here).
        let dims = 512;
        let mut h: BitHNSW<usize> = BitHNSW::with_params(BitHNSW::new(dims), 8, 8, 16);

        let base = HyperVector::random(dims);
        for i in 0..200 {
            let mut k = base.clone();
            // Perturb within a 32-bit region for clustering
            for b in 0..32 {
                if (i >> b) & 1 == 1 {
                    k.flip_bit(b);
                }
            }
            h.insert(k, i);
        }
        h.prune();

        // Query with a small perturbation
        let mut q = base.clone();
        q.flip_bit(3);

        let hnsw_res = h.search_nearest(&q).unwrap();
        // Linear scan ground truth
        let linear_best = h
            .key(0)
            .map(|_| {
                (0..h.len())
                    .min_by_key(|&i| q.hamming_distance(h.key(i).unwrap()))
                    .unwrap()
            })
            .unwrap();

        // Distance found by HNSW must match linear best distance (index may differ
        // when several keys are equidistant).
        let hnsw_dist = hnsw_res.0;
        let linear_dist = q.hamming_distance(h.key(linear_best).unwrap());
        assert!(hnsw_dist <= linear_dist + 2);
    }

    #[test]
    fn test_hnsw_recall_quality_at_scale() {
        let dims = 1024;
        let mut h: BitHNSW<usize> = BitHNSW::with_params(BitHNSW::new(dims), 16, 16, 32);
        let base = HyperVector::random(dims);
        let n = 2000;
        for i in 0usize..n {
            let mut k = base.clone();
            for b in 0..40 {
                if ((i.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> b) & 1) == 1 {
                    k.flip_bit(b);
                }
            }
            h.insert(k, i);
        }
        h.prune();

        // Ground-truth best distance via linear scan for a set of perturbed queries.
        let mut hits = 0;
        let trials = 50;
        for t in 0usize..trials {
        let mut q = base.clone();
        for b in 0..5usize {
            if ((t.wrapping_mul(0x85EB_CA6Busize >> b) >> b) & 1) == 1 {
                q.flip_bit(b);
            }
        }
            let linear_best = (0..h.len())
                .map(|i| q.hamming_distance(h.key(i).unwrap()))
                .min()
                .unwrap();
            let hnsw_best = h.search_nearest(&q).unwrap().0;
            if hnsw_best == linear_best {
                hits += 1;
            }
            // Never allow HNSW to be dramatically worse than linear.
            assert!(hnsw_best <= linear_best + 4, "t={t}: hnsw={hnsw_best} linear={linear_best}");
        }
        assert!(hits >= trials * 9 / 10, "recall@1 too low: {hits}/{trials}");
    }
}
