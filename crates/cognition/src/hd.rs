use crate::HyperVector;

/// Bundle (majority sum) a set of hypervectors into a single prototype.
///
/// Each bit of the result is the majority vote across all input vectors.
/// For even-length inputs, ties default to 0.
///
/// This is the "addition" operation in hyperdimensional computing.
/// It produces a vector that is similar to all input vectors.
pub fn bundle(vectors: &[&HyperVector]) -> HyperVector {
    assert!(!vectors.is_empty(), "bundle requires at least one vector");
    let dims = vectors[0].dims();
    for v in &vectors[1..] {
        assert_eq!(v.dims(), dims, "dimension mismatch in bundle");
    }

    let words = dims.div_ceil(64);
    let n = vectors.len() as i64;

    let mut result = HyperVector::new(dims);

    for w in 0..words {
        let mut counts: i64 = 0;
        for b in 0..64.min(dims - w * 64) {
            let bit_idx = w * 64 + b;

            for v in vectors {
                if v.get_bit(bit_idx) {
                    counts += 1;
                } else {
                    counts -= 1;
                }
            }

            if counts > 0 || (counts == 0 && n % 2 == 0) {
                result.set_bit(bit_idx, true);
            }
            counts = 0;
        }
    }

    result
}

/// Bind (XOR) two hypervectors to form an association.
///
/// Binding is its own inverse: bind(a, bind(a, b)) == b.
/// This is used to associate two items or to encode relationships.
pub fn bind(a: &HyperVector, b: &HyperVector) -> HyperVector {
    a.xor(b)
}

/// Permute (rotate) a hypervector by `shift` positions.
///
/// Permutation is used to encode order or position,
/// e.g., converting a sequence into a single hypervector.
pub fn permute(v: &HyperVector, shift: usize) -> HyperVector {
    v.permute(shift)
}

/// Encode a sequence of hypervectors into a single hypervector
/// by permuting each item by its position and bundling (majority vote).
///
/// This preserves sequence order information while staying
/// similarity-preserving: sequences that share items in similar positions
/// encode to nearby vectors.
///
/// For example, the sequence [A, B, C] becomes:
///   bundle(permute(A, 1), permute(B, 2), permute(C, 3))
pub fn encode_sequence(sequence: &[&HyperVector]) -> HyperVector {
    assert!(
        !sequence.is_empty(),
        "encode_sequence requires at least one item"
    );
    let permuted: Vec<HyperVector> = sequence
        .iter()
        .enumerate()
        .map(|(pos, &item)| permute(item, pos + 1))
        .collect();
    let refs: Vec<&HyperVector> = permuted.iter().collect();
    bundle(&refs)
}

/// N-gram bundling: bundle each item with permuted neighbors.
///
/// This encodes local context windows, similar to how n-gram
/// language models capture local patterns.
pub fn encode_ngrams(sequence: &[&HyperVector], n: usize) -> Vec<HyperVector> {
    if sequence.len() < n {
        return Vec::new();
    }

    let mut ngrams = Vec::with_capacity(sequence.len() - n + 1);
    for window in sequence.windows(n) {
        let mut acc = window[0].clone();
        for (offset, &item) in window.iter().enumerate().skip(1) {
            acc = bind(&acc, &permute(item, offset));
        }
        ngrams.push(acc);
    }
    ngrams
}

/// Graph encoding: encode a directed edge between two nodes.
///
/// edge(a, b) = bind(permute(a, 1), bind(permute(b, 2), role_vector))
/// where role_vector distinguishes this type of relationship.
pub fn encode_edge(from: &HyperVector, to: &HyperVector, role: &HyperVector) -> HyperVector {
    let a = permute(from, 1);
    let b = permute(to, 2);
    let ab = bind(&a, &b);
    bind(&ab, role)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bundle_same_vectors() {
        let v = HyperVector::random(256);
        let result = bundle(&[&v, &v, &v]);
        let sim = v.similarity(&result);
        assert!(
            sim > 0.99,
            "bundle of identical vectors should be very similar: {}",
            sim
        );
    }

    #[test]
    fn test_bundle_orthogonal_vectors() {
        let a = HyperVector::from_bits(&[true, true, false, false]);
        let b = HyperVector::from_bits(&[true, false, true, false]);
        let c = HyperVector::from_bits(&[false, true, false, true]);
        let result = bundle(&[&a, &b, &c]);
        // a[0]=T, b[0]=T, c[0]=F -> majority T
        assert!(result.get_bit(0));
        // a[1]=T, b[1]=F, c[1]=T -> majority T
        assert!(result.get_bit(1));
        // a[2]=F, b[2]=T, c[2]=F -> majority F
        assert!(!result.get_bit(2));
        // a[3]=F, b[3]=F, c[3]=T -> majority F
        assert!(!result.get_bit(3));
    }

    #[test]
    fn test_bind_self_inverse() {
        let a = HyperVector::random(128);
        let b = HyperVector::random(128);
        let bound = bind(&a, &b);
        let recovered = bind(&bound, &b);
        assert_eq!(recovered.hamming_distance(&a), 0);
    }

    #[test]
    fn test_permute_cyclic() {
        let a = HyperVector::from_bits(&[true, false, false, false]);
        let p1 = permute(&a, 1);
        assert!(p1.get_bit(1));
        let p2 = permute(&p1, 3);
        assert!(p2.get_bit(0));
    }

    #[test]
    fn test_encode_sequence() {
        let a = HyperVector::random(128);
        let b = HyperVector::random(128);
        let seq1 = encode_sequence(&[&a, &b]);
        let seq2 = encode_sequence(&[&b, &a]);
        // Different orderings should give different results
        assert!(seq1.similarity(&seq2) < 0.8);
    }

    #[test]
    fn test_encode_ngrams() {
        let a = HyperVector::random(64);
        let b = HyperVector::random(64);
        let c = HyperVector::random(64);
        let ngrams = encode_ngrams(&[&a, &b, &c], 2);
        assert_eq!(ngrams.len(), 2);
    }

    #[test]
    fn test_encode_edge() {
        let from = HyperVector::random(64);
        let to = HyperVector::random(64);
        let role = HyperVector::random(64);
        let edge = encode_edge(&from, &to, &role);
        assert_eq!(edge.dims(), 64);
    }
}
