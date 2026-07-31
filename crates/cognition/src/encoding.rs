use bitllm_tensor::pnword::PNActivation256;
use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::HyperVector;

/// Trit positions in a `PNActivation256`.
const NUM_TRITS: usize = 128;

/// Direct trit-preserving mapping: each 2-bit trit becomes 2 HyperVector bits.
///
/// `+1` -> `01`, `-1` -> `10`, `0` -> `00`. Hamming distance in HyperVector
/// space is then exactly the trit distance scaled (0 vs `+1`/`-1` = 1 bit,
/// `+1` vs `-1` = 2 bits). This is lossless, but for sparse activations the
/// result is mostly zeros, which degrades the HD algebra (`bundle`'s majority
/// vote collapses toward all-zero).
pub fn encode_activation_direct(pn: &PNActivation256) -> HyperVector {
    HyperVector::from_words(pn.trits.to_vec(), NUM_TRITS * 2)
}

/// Random-indexing codebook: maps each (position, sign) to a random balanced
/// HyperVector, combining the active codes by **bundle** (majority vote).
///
/// Each active code is balanced, so the majority vote of any number of them
/// stays ~0.5-density: even a very sparse activation encodes to a dense key,
/// keeping `bundle`/`permute`/`encode_sequence` well-behaved.
///
/// Unlike XOR (bind), bundle does *not* cancel shared codes — bind's
/// self-inverse property means XOR-combined keys measure the symmetric
/// difference, so overlap similarity is destroyed. Bundle preserves it:
/// Hamming distance approximates activation overlap (the shared set of active
/// position-sign pairs), which is what retrieval needs.
#[derive(Debug, Clone)]
pub struct RandomIndexCodebook {
    dims: usize,
    /// `codes[pos * 2 + sign]`, sign 0 = +1, 1 = -1.
    codes: Vec<HyperVector>,
    /// Fixed random tie-break bits for majority votes with even counts.
    tie: Vec<u64>,
}

impl RandomIndexCodebook {
    pub fn new(dims: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(0x5EED_CAFE);
        let mut codes = Vec::with_capacity(NUM_TRITS * 2);
        for _ in 0..NUM_TRITS * 2 {
            codes.push(HyperVector::random_with(dims, &mut rng));
        }
        let words = dims.div_ceil(64);
        let mut tie = Vec::with_capacity(words);
        for _ in 0..words {
            tie.push(rng.gen());
        }
        Self { dims, codes, tie }
    }

    pub fn dims(&self) -> usize {
        self.dims
    }

    pub fn encode(&self, pn: &PNActivation256) -> HyperVector {
        let mut active: Vec<&HyperVector> = Vec::with_capacity(pn.popcount() as usize);
        for w in 0..4 {
            let mut bits = pn.trits[w];
            while bits != 0 {
                let tz = bits.trailing_zeros() as usize;
                let trit_idx = w * 32 + tz / 2;
                let sign = tz & 1; // even = positive, odd = negative
                active.push(&self.codes[trit_idx * 2 + sign]);
                bits &= !(0b11u64 << (tz & !1));
            }
        }
        match active.len() {
            0 => HyperVector::new(self.dims),
            1 => active[0].clone(),
            _ => majority_vote(active, &self.tie, self.dims),
        }
    }
}

/// Per-bit majority vote over `codes`. Ties (even counts) resolve via the
/// fixed random `tie` bits, keeping the result balanced at ~0.5 density for
/// any number of codes. Works on `HyperVector` word slices via the public API.
fn majority_vote(codes: Vec<&HyperVector>, tie: &[u64], dims: usize) -> HyperVector {
    let k = codes.len() as i64;
    let words = dims.div_ceil(64);
    let mut counts = vec![0i64; words * 64];
    for v in codes {
        let slice = v.as_slice();
        for (w, &word) in slice.iter().enumerate() {
            let base = w * 64;
            for b in 0..64 {
                if (word >> b) & 1 == 1 {
                    counts[base + b] += 1;
                }
            }
        }
    }
    let mut data = vec![0u64; words];
    for (i, &c) in counts.iter().enumerate() {
        let w = i / 64;
        let b = i % 64;
        let on = if c * 2 > k {
            true
        } else if c * 2 == k {
            (tie[w] >> b) & 1 == 1
        } else {
            false
        };
        if on {
            data[w] |= 1u64 << b;
        }
    }
    HyperVector::from_words(data, dims)
}

/// Encode a sequence of activation keys with position information
/// (permute by index, then bundle).
pub fn encode_sequence_positional(seq: &[&HyperVector]) -> HyperVector {
    crate::hd::encode_sequence(seq)
}

/// Encode a sequence of activation keys position-free (plain bundle).
pub fn encode_sequence_plain(seq: &[&HyperVector]) -> HyperVector {
    crate::hd::bundle(seq)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sparse_packet(nonzero: usize, seed: u64) -> PNActivation256 {
        let mut values = [0i8; 128];
        let mut rng = StdRng::seed_from_u64(seed);
        for _ in 0..nonzero {
            let i = rng.gen_range(0..128);
            values[i] = if rng.gen_bool(0.5) { 1 } else { -1 };
        }
        PNActivation256::pack(&values)
    }

    #[test]
    fn test_direct_preserves_trit_distance() {
        let mut a_vals = [0i8; 128];
        a_vals[0] = 1;
        a_vals[10] = -1;
        let a = PNActivation256::pack(&a_vals);

        let mut b_vals = a_vals;
        b_vals[10] = 1; // same trit, opposite sign: +1 vs -1 = 2 bits
        let b = PNActivation256::pack(&b_vals);

        let ha = encode_activation_direct(&a);
        let hb = encode_activation_direct(&b);
        assert_eq!(ha.hamming_distance(&hb), 2);

        let mut c_vals = a_vals;
        c_vals[5] = -1; // new active trit where a had 0: 0 vs -1 = 1 bit
        let c = PNActivation256::pack(&c_vals);
        assert_eq!(ha.hamming_distance(&encode_activation_direct(&c)), 1);

        // Identical packets map to identical keys.
        assert_eq!(ha.hamming_distance(&encode_activation_direct(&a)), 0);
    }

    #[test]
    fn test_random_index_is_dense_and_deterministic() {
        let cb = RandomIndexCodebook::new(512);
        let sparse = sparse_packet(8, 42); // 6% density
        let hv = cb.encode(&sparse);
        assert_eq!(hv.dims(), 512);
        let density = hv.density();
        assert!(
            (0.40..=0.60).contains(&density),
            "random-indexed sparse activation should be balanced, got density {density}"
        );
        assert_eq!(cb.encode(&sparse).hamming_distance(&hv), 0, "encoding must be deterministic");
    }

    #[test]
    fn test_random_index_preserves_overlap() {
        let cb = RandomIndexCodebook::new(512);
        // Two packets sharing half their active trits.
        let mut a_vals = [0i8; 128];
        let mut b_vals = [0i8; 128];
        for i in 0..16 {
            a_vals[i] = if i % 2 == 0 { 1 } else { -1 };
            if i % 2 == 0 {
                b_vals[i] = a_vals[i]; // shared
            } else {
                b_vals[64 + i] = a_vals[i]; // moved elsewhere
            }
        }
        let a = PNActivation256::pack(&a_vals);
        let b = PNActivation256::pack(&b_vals);
        // Fully disjoint reference packet.
        let mut c_vals = [0i8; 128];
        for i in 0..16 {
            c_vals[80 + i] = if i % 2 == 0 { 1 } else { -1 };
        }
        let c = PNActivation256::pack(&c_vals);

        let (ha, hb, hc) = (cb.encode(&a), cb.encode(&b), cb.encode(&c));
        let ab = ha.hamming_distance(&hb);
        let ac = ha.hamming_distance(&hc);
        assert!(ab < ac, "overlapping activations must be closer: ab={ab} ac={ac}");
    }

    #[test]
    fn test_sequence_position_information() {
        // Positional encoding should separate different orderings of the same
        // packets; plain bundle should not (much).
        let cb = RandomIndexCodebook::new(512);
        let p1 = cb.encode(&sparse_packet(10, 1));
        let p2 = cb.encode(&sparse_packet(10, 2));
        let p3 = cb.encode(&sparse_packet(10, 3));

        let seq_abc = encode_sequence_positional(&[&p1, &p2, &p3]);
        let seq_cba = encode_sequence_positional(&[&p3, &p2, &p1]);
        assert!(
            seq_abc.similarity(&seq_cba) < 0.75,
            "positional encoding should separate orderings: {}",
            seq_abc.similarity(&seq_cba)
        );

        let plain_abc = encode_sequence_plain(&[&p1, &p2, &p3]);
        let plain_cba = encode_sequence_plain(&[&p3, &p2, &p1]);
        assert!(
            plain_abc.similarity(&plain_cba) > 0.9,
            "plain bundle should be order-insensitive: {}",
            plain_abc.similarity(&plain_cba)
        );
    }

    #[test]
    fn test_packet_retrieval_via_linear_recall() {
        // End-to-end: codebook-encode stored packets, probe with a noisy
        // version sharing 75% of activations, and recover the original via
        // exact linear recall. Independent random packets — no key locality —
        // are exactly the regime where a flat scan is the correct tool.
        use crate::SparseAssociativeMemory;

        let n = 512;
        let cb = RandomIndexCodebook::new(512);
        let packets: Vec<PNActivation256> = (0..n).map(|i| sparse_packet(16, i as u64 + 1)).collect();

        let mut mem: SparseAssociativeMemory<usize> = SparseAssociativeMemory::with_capacity(n);
        for (i, p) in packets.iter().enumerate() {
            mem.store(cb.encode(p), i);
        }

        let mut hits = 0usize;
        for (i, p) in packets.iter().enumerate() {
            let probe = cb.encode(&noisy_probe(p, 0.75, 0xDEAD_BEEF + i as u64));
            if mem.recall(&probe).map(|(_, idx)| *idx) == Some(i) {
                hits += 1;
            }
        }
        let recall = hits as f32 / n as f32;
        assert!(recall >= 0.9, "linear recall must recover original packet, recall@1 = {recall}");
    }

    // NOTE: graph-based retrieval of these same encoded keys was attempted
    // (`test_packet_retrieval_via_hnsw`) and failed for every key distribution
    // tried: independent packets have no locality, shared-topic packets
    // collapse into a dense ball, and even a Gray-code lattice was not
    // navigated because single-layer NSW lacks long-range edges. See the
    // KNOWN LIMITATION note on `BitHNSW`. Flat linear recall is the working
    // path for packet keys today.

    #[test]
    fn test_direct_vs_random_index_distance() {
        // Two packets sharing half their active trits. Direct encoding gives
        // exact trit distance; random-index bundle should give a *lower*
        // distance (higher similarity) than for a disjoint packet.
        let cb = RandomIndexCodebook::new(512);
        let mut a_vals = [0i8; 128];
        let mut b_vals = [0i8; 128];
        for i in 0..16 {
            a_vals[i] = if i % 2 == 0 { 1 } else { -1 };
            b_vals[i] = a_vals[i]; // full overlap here
        }
        for i in 0..16 {
            b_vals[64 + i] = if i % 2 == 0 { 1 } else { -1 };
        }
        let a = PNActivation256::pack(&a_vals);
        let b = PNActivation256::pack(&b_vals);

        let mut c_vals = [0i8; 128];
        for i in 0..16 {
            c_vals[80 + i] = if i % 2 == 0 { 1 } else { -1 };
        }
        let c = PNActivation256::pack(&c_vals);

        let (ha, hb, hc) = (cb.encode(&a), cb.encode(&b), cb.encode(&c));
        let ab = ha.similarity(&hb);
        let ac = ha.similarity(&hc);
        assert!(
            ab > ac + 0.1,
            "random-index bundle must preserve overlap similarity: ab={ab} ac={ac}"
        );
    }

    /// Probe keeping `keep_frac` of a packet's activations, refilling the
    /// dropped ones with fresh random activations (new evidence). Total active
    /// count is preserved, so it stays a realistic sparse packet.
    fn noisy_probe(p: &PNActivation256, keep_frac: f64, seed: u64) -> PNActivation256 {
        let mut values = [0i8; 128];
        let mut rng = StdRng::seed_from_u64(seed);
        p.unpack(&mut values);
        let mut total = 0usize;
        let mut kept = 0usize;
        for v in values.iter_mut() {
            if *v != 0 {
                total += 1;
                if rng.gen_bool(keep_frac) {
                    kept += 1;
                } else {
                    *v = 0;
                }
            }
        }
        let mut refilled = 0usize;
        while refilled < total - kept {
            let i = rng.gen_range(0..128);
            if values[i] == 0 {
                values[i] = if rng.gen_bool(0.5) { 1 } else { -1 };
                refilled += 1;
            }
        }
        PNActivation256::pack(&values)
    }
}
