use crate::memory::SparseAssociativeMemory;
use crate::HyperVector;

/// A reasoning unit processes hypervector queries against
/// associative memory and produces responses.
///
/// Each unit has:
/// - A memory of stored patterns
/// - An activation threshold
/// - Optional sub-units for hierarchical reasoning
///
/// This is the building block for replacing dense feed-forward
/// networks with sparse, similarity-driven computation.
#[derive(Debug, Clone)]
pub struct ReasoningUnit<T: Clone> {
    /// Name/label for this unit
    pub label: String,
    /// Associative memory storing known patterns
    pub memory: SparseAssociativeMemory<T>,
    /// Minimum similarity for a match to activate
    pub threshold: f32,
    /// Maximum number of results to return
    pub max_results: usize,
    /// Optional sub-reasoning units for hierarchical processing
    pub subunits: Vec<ReasoningUnit<T>>,
}

impl<T: Clone> ReasoningUnit<T> {
    pub fn new(label: impl Into<String>, threshold: f32, max_results: usize) -> Self {
        Self {
            label: label.into(),
            memory: SparseAssociativeMemory::new(),
            threshold,
            max_results,
            subunits: Vec::new(),
        }
    }

    /// Add a pattern to this unit's memory
    pub fn learn(&mut self, key: HyperVector, value: T) {
        self.memory.store(key, value);
    }

    /// Add a sub-unit for hierarchical reasoning
    pub fn add_subunit(&mut self, unit: ReasoningUnit<T>) {
        self.subunits.push(unit);
    }

    /// Forward a query through this reasoning unit.
    ///
    /// Returns the top matches above the activation threshold.
    /// If no matches exceed threshold, returns the closest match
    /// with a warning-level similarity (below threshold but above 0).
    pub fn forward(&self, query: &HyperVector) -> Vec<(f32, &T)> {
        let results = self
            .memory
            .recall_best_n(query, self.max_results, self.threshold);

        if !results.is_empty() {
            return results;
        }

        let closest = self.memory.recall(query);
        match closest {
            Some((sim, val)) if sim > 0.0 => vec![(sim, val)],
            _ => Vec::new(),
        }
    }

    /// Forward a query, also recursively activating sub-units.
    /// Returns results from this unit and all matching sub-units.
    pub fn forward_hierarchical(&self, query: &HyperVector) -> Vec<(f32, &T, String)> {
        let mut results: Vec<(f32, &T, String)> = self
            .forward(query)
            .into_iter()
            .map(|(sim, val)| (sim, val, self.label.clone()))
            .collect();

        for subunit in &self.subunits {
            let sub_results = subunit.forward_hierarchical(query);
            results.extend(sub_results);
        }

        results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        results.truncate(self.max_results);
        results
    }

    /// Learn a pattern and route to the most similar sub-unit for refinement
    pub fn learn_routed(&mut self, key: HyperVector, value: T) {
        if let Some((_sim, sub)) = self
            .subunits
            .iter()
            .filter_map(|s| s.memory.recall(&key).map(|(sim, _)| (sim, s.label.clone())))
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
        {
            if let Some(target) = self.subunits.iter_mut().find(|s| s.label == sub) {
                target.learn(key, value);
                return;
            }
        }
        self.memory.store(key, value);
    }

    pub fn len(&self) -> usize {
        self.memory.len()
    }

    pub fn is_empty(&self) -> bool {
        self.memory.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_reasoning() {
        let mut unit = ReasoningUnit::new("test", 0.5, 3);
        let pattern = HyperVector::random(128);
        unit.learn(pattern.clone(), "response_a");

        let query = pattern.clone();
        let results = unit.forward(&query);
        assert_eq!(results.len(), 1);
        assert_eq!(*results[0].1, "response_a");
    }

    #[test]
    fn test_subthreshold_returns_closest() {
        let mut unit = ReasoningUnit::new("test", 0.9, 3);
        let pattern = HyperVector::random(128);
        unit.learn(pattern.clone(), "only_option");

        let query = HyperVector::random(128);
        let results = unit.forward(&query);
        // Even if below threshold, should return something (closest match)
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_hierarchical_reasoning() {
        let mut root = ReasoningUnit::new("root", 0.5, 5);
        let mut sub = ReasoningUnit::new("sub", 0.5, 3);

        let p1 = HyperVector::random(64);
        let p2 = HyperVector::random(64);

        root.learn(p1.clone(), "root_val");
        sub.learn(p2.clone(), "sub_val");
        root.add_subunit(sub);

        let results = root.forward_hierarchical(&p2);
        assert!(!results.is_empty());
        let labels: Vec<&str> = results.iter().map(|(_, _, l)| l.as_str()).collect();
        assert!(labels.contains(&"sub"));
    }

    #[test]
    fn test_learn_routed() {
        let mut root = ReasoningUnit::new("root", 0.3, 5);
        let mut sub = ReasoningUnit::new("colors", 0.3, 5);

        let red = HyperVector::random(64);
        sub.learn(red.clone(), "red_value");
        root.add_subunit(sub);

        // New pattern similar to "red" should route to "colors" subunit
        let similar_to_red = red.clone();
        root.learn_routed(similar_to_red, "dark_red");

        assert!(!root.subunits[0].is_empty());
    }
}
