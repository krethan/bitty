use crate::HyperVector;

/// A routing pathway that defines one branch of computation.
#[derive(Debug, Clone)]
pub struct Pathway {
    /// Label for this pathway
    pub label: String,
    /// Key vector: queries similar to this are routed here
    pub key: HyperVector,
    /// Routing weight (amplifies or dampens similarity)
    pub weight: f32,
}

/// Similarity-based router that replaces dense matrix multiplication
/// with sparse, content-addressed routing.
///
/// Instead of computing `output = input @ W`, the router computes:
/// `routing = route(input, pathways)` then activates only the matched
/// pathways. This is the core idea behind routing-based execution
/// as an alternative to feed-forward networks.
#[derive(Debug, Clone)]
pub struct Router {
    pathways: Vec<Pathway>,
    /// If true, soft-routing: all pathways contribute weighted by similarity.
    /// If false, hard-routing: only the top pathway activates.
    pub soft_routing: bool,
    /// Top-k pathways to consider in hard-routing mode
    pub top_k: usize,
    /// Minimum similarity threshold
    pub threshold: f32,
}

impl Router {
    pub fn new() -> Self {
        Self {
            pathways: Vec::new(),
            soft_routing: true,
            top_k: 1,
            threshold: 0.0,
        }
    }

    /// Add a routing pathway
    pub fn add_pathway(&mut self, label: impl Into<String>, key: HyperVector, weight: f32) {
        self.pathways.push(Pathway {
            label: label.into(),
            key,
            weight,
        });
    }

    /// Route a query to the most similar pathways.
    ///
    /// Returns `(pathway_index, activation_strength)` pairs sorted by
    /// activation strength descending.
    pub fn route(&self, query: &HyperVector) -> Vec<(usize, f32)> {
        let mut activations: Vec<(usize, f32)> = self
            .pathways
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let sim = query.similarity(&p.key);
                let activation = sim * p.weight;
                (i, activation)
            })
            .filter(|(_, a)| *a >= self.threshold)
            .collect();

        activations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        if !self.soft_routing {
            activations.truncate(self.top_k);
        }

        activations
    }

    /// Hard-route: return at most one pathway (the best match)
    pub fn route_hard(&self, query: &HyperVector) -> Option<(usize, f32)> {
        self.route(query).into_iter().next()
    }

    /// Get a reference to a pathway by index
    pub fn get_pathway(&self, idx: usize) -> Option<&Pathway> {
        self.pathways.get(idx)
    }

    /// Number of pathways
    pub fn num_pathways(&self) -> usize {
        self.pathways.len()
    }

    /// Check if any pathway would activate for a given query
    pub fn would_activate(&self, query: &HyperVector) -> bool {
        self.pathways
            .iter()
            .any(|p| query.similarity(&p.key) * p.weight >= self.threshold)
    }
}

impl Default for Router {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_route_basic() {
        let mut router = Router::new();
        let k1 = HyperVector::random(128);
        let k2 = HyperVector::random(128);

        router.add_pathway("path_a", k1.clone(), 1.0);
        router.add_pathway("path_b", k2.clone(), 1.0);

        let results = router.route(&k1);
        assert!(!results.is_empty());
        assert_eq!(router.get_pathway(results[0].0).unwrap().label, "path_a");
    }

    #[test]
    fn test_hard_route_returns_only_top() {
        let mut router = Router::new();
        router.soft_routing = false;
        router.top_k = 1;

        let query = HyperVector::random(64);
        let similar = query.clone();
        let different = HyperVector::random(64);

        router.add_pathway("close", similar.clone(), 1.0);
        router.add_pathway("far", different, 0.1);

        let result = router.route_hard(&query);
        assert!(result.is_some());
        assert_eq!(
            router.get_pathway(result.unwrap().0).unwrap().label,
            "close"
        );
    }

    #[test]
    fn test_threshold_filters() {
        let mut router = Router::new();
        router.threshold = 0.8;

        let query = HyperVector::random(64);
        let similar = query.clone();
        let mut different = query.clone();
        for i in 0..32 {
            different.flip_bit(i);
        }

        router.add_pathway("close", similar.clone(), 1.0);
        router.add_pathway("far", different, 1.0);

        let results = router.route(&query);
        for (_, activation) in &results {
            assert!(*activation >= 0.8);
        }
    }

    #[test]
    fn test_weight_scales_activation() {
        let mut router = Router::new();
        let k = HyperVector::random(64);
        let query = k.clone();

        router.add_pathway("weighted", k.clone(), 2.0);

        let results = router.route(&query);
        let (_, activation) = results[0];
        assert!((activation - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_empty_router() {
        let router: Router = Router::new();
        let query = HyperVector::random(64);
        assert!(router.route(&query).is_empty());
        assert!(router.route_hard(&query).is_none());
    }

    #[test]
    fn test_would_activate() {
        let mut router = Router::new();
        router.threshold = 0.5;
        let k = HyperVector::random(64);
        router.add_pathway("test", k.clone(), 1.0);
        assert!(router.would_activate(&k));
    }
}
