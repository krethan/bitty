use crate::HyperVector;

/// Extracts binary hypervectors from f32 embeddings.
///
/// Uses random projections (locality-sensitive hashing) to map
/// continuous embeddings into binary hypervectors while preserving
/// similarity structure.
///
/// This is the bridge between the existing transformer-based
/// embedding space and the binary hypervector space.
#[derive(Debug, Clone)]
pub struct FeatureExtractor {
    /// Output dimensionality in bits
    output_dims: usize,
    /// Random projection matrix: output_dims vectors of input_dims bits each
    /// Stored as a vec of HyperVectors (each representing one random projection)
    projections: Vec<HyperVector>,
    /// Optional bias for the projections
    bias: Option<Vec<f32>>,
}

impl FeatureExtractor {
    /// Create a new feature extractor with random projections.
    ///
    /// `input_dim`: dimensionality of incoming f32 embeddings
    /// `output_dim`: desired binary hypervector dimensionality (bits)
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        let projections = (0..output_dim)
            .map(|_| HyperVector::random(input_dim))
            .collect();
        Self {
            output_dims: output_dim,
            projections,
            bias: None,
        }
    }

    /// Create a feature extractor with a learned projection and bias.
    /// `weights` should be shape [output_dim, input_dim] flattened, values in [-1, 1]
    /// `bias_val` is optional per-output bias
    pub fn with_weights(input_dim: usize, output_dim: usize, weights: &[f32], bias_val: Option<&[f32]>) -> Self {
        assert_eq!(weights.len(), output_dim * input_dim, "weight matrix size mismatch");
        if let Some(b) = bias_val {
            assert_eq!(b.len(), output_dim, "bias size mismatch");
        }

        let projections: Vec<HyperVector> = weights
            .chunks(input_dim)
            .map(HyperVector::from_f32_slice)
            .collect();

        Self {
            output_dims: output_dim,
            projections,
            bias: bias_val.map(|b| b.to_vec()),
        }
    }

    pub fn output_dims(&self) -> usize {
        self.output_dims
    }

    /// Encode a single f32 embedding into a binary hypervector.
    ///
    /// Each output bit is the sign of the dot product between the
    /// input and a random projection vector, optionally with bias.
    pub fn encode(&self, embedding: &[f32]) -> HyperVector {
        assert_eq!(
            embedding.len(),
            self.projections[0].dims(),
            "embedding dimension mismatch"
        );
        let mut result = HyperVector::new(self.output_dims);
        for (i, proj) in self.projections.iter().enumerate() {
            let dot: f32 = embedding
                .iter()
                .enumerate()
                .map(|(j, &v)| if proj.get_bit(j) { v } else { -v })
                .sum();
            let biased = dot + self.bias.as_ref().map_or(0.0, |b| b[i]);
            if biased > 0.0 {
                result.set_bit(i, true);
            }
        }
        result
    }

    /// Encode a batch of embeddings into binary hypervectors.
    pub fn encode_batch(&self, embeddings: &[&[f32]]) -> Vec<HyperVector> {
        embeddings.iter().map(|e| self.encode(e)).collect()
    }

    /// Compute the similarity between two f32 embeddings in hypervector space
    /// without materializing the full hypervectors (projection + compare).
    pub fn compare(&self, a: &[f32], b: &[f32]) -> f32 {
        let hv_a = self.encode(a);
        let hv_b = self.encode(b);
        hv_a.similarity(&hv_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_consistent() {
        let extractor = FeatureExtractor::new(32, 128);
        let emb = vec![0.5; 32];
        let hv1 = extractor.encode(&emb);
        let hv2 = extractor.encode(&emb);
        assert_eq!(hv1.hamming_distance(&hv2), 0);
    }

    #[test]
    fn test_similar_inputs_produce_similar_vectors() {
        let extractor = FeatureExtractor::new(16, 64);
        let a = vec![0.8; 16];
        let mut b = a.clone();
        b[0] = 0.9;
        let hv_a = extractor.encode(&a);
        let hv_b = extractor.encode(&b);
        let sim = hv_a.similarity(&hv_b);
        assert!(sim > 0.5, "similar inputs should have similar hypervectors: {}", sim);
    }

    #[test]
    fn test_different_inputs_produce_different_vectors() {
        let extractor = FeatureExtractor::new(16, 128);
        let a = vec![1.0; 16];
        let b = vec![-1.0; 16];
        let hv_a = extractor.encode(&a);
        let hv_b = extractor.encode(&b);
        let sim = hv_a.similarity(&hv_b);
        assert!(sim < 0.5, "opposite inputs should have different hypervectors: {}", sim);
    }

    #[test]
    fn test_encode_batch() {
        let extractor = FeatureExtractor::new(8, 16);
        let embeddings = [vec![0.1; 8], vec![0.2; 8], vec![0.3; 8]];
        let refs: Vec<&[f32]> = embeddings.iter().map(|v| v.as_slice()).collect();
        let result = extractor.encode_batch(&refs);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].dims(), 16);
    }

    #[test]
    fn test_with_weights() {
        let input_dim = 4;
        let output_dim = 3;
        let weights = vec![1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0];
        let bias = vec![0.0, 0.0, 0.0];
        let extractor = FeatureExtractor::with_weights(input_dim, output_dim, &weights, Some(&bias));
        assert_eq!(extractor.output_dims(), 3);
        let emb = vec![0.5, -0.5, 0.5, -0.5];
        let hv = extractor.encode(&emb);
        assert_eq!(hv.dims(), 3);
    }
}
