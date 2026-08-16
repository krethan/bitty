use std::collections::{HashMap, VecDeque};

use bitllm_tensor::pnword::{PNActivation256, PNWeight256};

/// Unique node identifier in the execution graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

/// A self-describing computational packet: packed neural data + routing metadata.
#[derive(Debug, Clone)]
pub struct Packet {
    pub activation: PNActivation256,
    pub destination: NodeId,
    pub priority: u8,
    pub ttl: u8,
    pub timestamp: u64,
}

static PACKET_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl Packet {
    pub fn new(activation: PNActivation256, destination: NodeId) -> Self {
        Self {
            activation,
            destination,
            priority: 0,
            ttl: 16,
            timestamp: PACKET_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        }
    }

    pub fn is_alive(&self) -> bool {
        self.ttl > 0
    }
}

/// A processing node in the execution graph.
pub trait Node: std::fmt::Debug + Send {
    fn id(&self) -> NodeId;
    fn process(&self, packet: &Packet) -> Vec<Packet>;
    fn name(&self) -> &str;
}

/// Directed execution graph connecting nodes.
#[derive(Debug)]
pub struct Graph {
    nodes: HashMap<NodeId, Box<dyn Node>>,
    adjacency: HashMap<NodeId, Vec<NodeId>>,
    next_id: u32,
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl Graph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            adjacency: HashMap::new(),
            next_id: 0,
        }
    }

    pub fn add_node(&mut self, node: Box<dyn Node>) -> NodeId {
        let id = NodeId(self.next_id);
        self.next_id += 1;
        self.nodes.insert(id, node);
        self.adjacency.entry(id).or_default();
        id
    }

    pub fn add_edge(&mut self, from: NodeId, to: NodeId) {
        self.adjacency.entry(from).or_default().push(to);
    }

    pub fn process(&self, packet: &Packet) -> Vec<Packet> {
        if let Some(node) = self.nodes.get(&packet.destination) {
            node.process(packet)
        } else {
            vec![]
        }
    }

    pub fn successors(&self, node_id: &NodeId) -> Option<&Vec<NodeId>> {
        self.adjacency.get(node_id)
    }

    #[allow(dead_code)]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

/// Event-driven scheduler routing packets through a graph.
#[derive(Debug)]
pub struct Scheduler {
    pub queue: VecDeque<Packet>,
    pub graph: Graph,
    pub cycles: u64,
    pub packets_processed: u64,
    pub packets_dropped: u64,
}

impl Scheduler {
    pub fn new(graph: Graph) -> Self {
        Self {
            queue: VecDeque::new(),
            graph,
            cycles: 0,
            packets_processed: 0,
            packets_dropped: 0,
        }
    }

    pub fn enqueue(&mut self, packet: Packet) {
        if packet.is_alive() {
            self.queue.push_back(packet);
        }
    }

    /// Dequeue one packet, process it through its destination node, route outputs.
    pub fn step(&mut self) -> usize {
        if let Some(packet) = self.queue.pop_front() {
            self.packets_processed += 1;
            let outputs = self.graph.process(&packet);
            for mut output in outputs {
                output.ttl = output.ttl.saturating_sub(1);
                if output.is_alive() {
                    self.queue.push_back(output);
                } else {
                    self.packets_dropped += 1;
                }
            }
            1
        } else {
            0
        }
    }

    /// Run until queue empty or max_steps reached.
    pub fn run(&mut self, max_steps: usize) -> usize {
        let mut steps = 0;
        while steps < max_steps && !self.queue.is_empty() {
            self.cycles += 1;
            steps += self.step();
        }
        steps
    }

    pub fn stats(&self) -> SchedulerStats {
        SchedulerStats {
            queue_depth: self.queue.len(),
            cycles: self.cycles,
            packets_processed: self.packets_processed,
            packets_dropped: self.packets_dropped,
        }
    }
}

/// Execution statistics for a scheduler run.
#[derive(Debug, Clone, PartialEq)]
pub struct SchedulerStats {
    pub queue_depth: usize,
    pub cycles: u64,
    pub packets_processed: u64,
    pub packets_dropped: u64,
}

// ---------------------------------------------------------------------------
// Built-in node implementations for benchmarks and testing
// ---------------------------------------------------------------------------

/// Node that performs a bitwise dot product against a fixed weight vector.
#[derive(Debug)]
pub struct MatMulNode {
    id: NodeId,
    name: String,
    weight: PNWeight256,
    successors: Vec<NodeId>,
}

impl MatMulNode {
    pub fn new(name: &str, weight: PNWeight256, successors: Vec<NodeId>) -> Self {
        Self {
            id: NodeId(0), // placeholder; overwritten by Graph::add_node
            name: name.to_string(),
            weight,
            successors,
        }
    }
}

impl Node for MatMulNode {
    fn id(&self) -> NodeId {
        self.id
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn process(&self, packet: &Packet) -> Vec<Packet> {
        let _ = packet.activation.dot(&self.weight);
        self.successors
            .iter()
            .map(|&succ| Packet::new(packet.activation, succ))
            .collect()
    }
}

/// Node that introduces sparsity by zeroing random trits.
#[derive(Debug)]
pub struct SparsifyNode {
    name: String,
    sparsity: f32,
    successors: Vec<NodeId>,
}

impl SparsifyNode {
    pub fn new(name: &str, sparsity: f32, successors: Vec<NodeId>) -> Self {
        Self {
            name: name.to_string(),
            sparsity,
            successors,
        }
    }
}

impl Node for SparsifyNode {
    fn id(&self) -> NodeId {
        NodeId(0)
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn process(&self, packet: &Packet) -> Vec<Packet> {
        let activation = packet.activation;
        let threshold = (self.sparsity * 128.0) as usize;

        let mask = !((1u128 << threshold) - 1);
        if (activation.active_mask & mask) == 0 && self.sparsity > 0.5 {
            // Return a dead packet so the scheduler drops it through TTL
            return self
                .successors
                .iter()
                .map(|&succ| {
                    let mut dead = Packet::new(activation, succ);
                    dead.ttl = 0;
                    dead
                })
                .collect();
        }

        self.successors
            .iter()
            .map(|&succ| Packet::new(activation, succ))
            .collect()
    }
}

/// Collects and counts received packets.
#[derive(Debug)]
pub struct SinkNode {
    name: String,
    pub received: std::sync::atomic::AtomicU64,
}

impl SinkNode {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            received: std::sync::atomic::AtomicU64::new(0),
        }
    }
}

impl Node for SinkNode {
    fn id(&self) -> NodeId {
        NodeId(0)
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn process(&self, _packet: &Packet) -> Vec<Packet> {
        self.received.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        vec![]
    }
}

impl Drop for SinkNode {
    fn drop(&mut self) {
        let _ = self.received.load(std::sync::atomic::Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_activation(active_count: usize) -> PNActivation256 {
        let mut vals = [0i8; 128];
        for (i, v) in vals.iter_mut().enumerate().take(active_count.min(128)) {
            *v = if i % 2 == 0 { 1 } else { -1 };
        }
        PNActivation256::pack(&vals)
    }

    fn make_weight() -> PNWeight256 {
        let vals: Vec<i8> = (0..128).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
        PNWeight256::pack(&vals, 1.0)
    }

    #[test]
    fn test_scheduler_single_node() {
        let mut graph = Graph::new();

        let sink_id = {
            let sink = Box::new(SinkNode::new("sink"));
            graph.add_node(sink)
        };

        let mut scheduler = Scheduler::new(graph);
        let packet = Packet::new(make_activation(4), sink_id);
        scheduler.enqueue(packet);
        scheduler.run(100);

        let stats = scheduler.stats();
        assert_eq!(stats.packets_processed, 1);
        assert_eq!(stats.queue_depth, 0);
    }

    #[test]
    fn test_scheduler_chain() {
        let mut graph = Graph::new();

        let w = make_weight();

        let sink_id = {
            let sink = Box::new(SinkNode::new("sink"));
            graph.add_node(sink)
        };

        let matmul_id = {
            let matmul = Box::new(MatMulNode::new("matmul", w, vec![sink_id]));
            graph.add_node(matmul)
        };

        graph.add_edge(matmul_id, sink_id);

        let mut scheduler = Scheduler::new(graph);
        let packet = Packet::new(make_activation(4), matmul_id);
        scheduler.enqueue(packet);
        scheduler.run(100);

        let stats = scheduler.stats();
        assert_eq!(stats.packets_processed, 2); // matmul + sink
    }

    #[test]
    fn test_scheduler_sparsity_skips_dead_packets() {
        let mut graph = Graph::new();

        let w = make_weight();
        let sparsity = 0.9;

        let sink_id = {
            let sink = Box::new(SinkNode::new("sink"));
            graph.add_node(sink)
        };

        let matmul_id = {
            let matmul = Box::new(MatMulNode::new("matmul", w, vec![sink_id]));
            graph.add_node(matmul)
        };

        let sparsify_id = {
            let sparsify = Box::new(SparsifyNode::new(
                "sparsify", sparsity, vec![matmul_id],
            ));
            graph.add_node(sparsify)
        };

        graph.add_edge(sparsify_id, matmul_id);
        graph.add_edge(matmul_id, sink_id);

        let mut scheduler = Scheduler::new(graph);
        for _ in 0..10 {
            let packet = Packet::new(make_activation(4), sparsify_id);
            scheduler.enqueue(packet);
        }
        scheduler.run(100);

        let stats = scheduler.stats();
        assert!(stats.packets_processed < 30);
        assert!(stats.packets_dropped > 0);
    }
}
