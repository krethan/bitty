use std::collections::VecDeque;

use crate::sampler::Sampler;
use crate::GpuContext;
use crate::Model;

/// A submitted generation request waiting for a free cache slot.
pub struct Request {
    pub id: u64,
    pub prompt: Vec<u32>,
    pub max_new_tokens: usize,
    pub eos: Option<u32>,
}

/// A finished generation result, drained by the caller.
#[derive(Debug, Clone, PartialEq)]
pub struct CompletedSequence {
    pub id: u64,
    pub tokens: Vec<u32>,
}

/// Per-slot state for a sequence currently being generated.
#[derive(Clone)]
struct ActiveSeq {
    id: u64,
    max_new_tokens: usize,
    eos: Option<u32>,
    generated: usize,
    output: Vec<u32>,
    logits: Vec<f32>,
}

/// Continuous batching scheduler.
///
/// Tracks a fixed pool of KV-cache slots. Requests are queued and scheduled
/// into the first free slot; a finished sequence (EOS or `max_new_tokens`)
/// frees its slot so a waiting request starts in the next step. New requests
/// therefore begin while others are still decoding — the defining property of
/// continuous batching (vs. lockstep `generate_batch`).
///
/// Each [`Model::continuous_step`] performs one scheduler iteration:
///   1. prefill queued requests into free slots,
///   2. sample one token per active slot,
///   3. decode the sampled tokens for still-running slots,
///   4. free finished slots and record their outputs.
pub struct ContinuousBatch {
    pub capacity: usize,
    active: Vec<Option<ActiveSeq>>,
    queue: VecDeque<Request>,
    next_id: u64,
    iterations: usize,
    completed: Vec<CompletedSequence>,
}

impl ContinuousBatch {
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            capacity,
            active: vec![None; capacity],
            queue: VecDeque::new(),
            next_id: 0,
            iterations: 0,
            completed: Vec::new(),
        }
    }

    /// Queue a new generation request. Returns its request id.
    pub fn submit(&mut self, prompt: &[u32], max_new_tokens: usize, eos: Option<u32>) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.queue.push_back(Request {
            id,
            prompt: prompt.to_vec(),
            max_new_tokens,
            eos,
        });
        id
    }

    /// Number of queued requests not yet scheduled into a slot.
    pub fn pending(&self) -> usize {
        self.queue.len()
    }

    /// Number of slots currently generating.
    pub fn active_count(&self) -> usize {
        self.active.iter().filter(|a| a.is_some()).count()
    }

    /// True while requests remain queued or slots are active.
    pub fn is_running(&self) -> bool {
        self.pending() > 0 || self.active_count() > 0
    }

    /// Scheduler iterations executed so far.
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// Drain completed sequences (in completion order).
    pub fn pop_completed(&mut self) -> Vec<CompletedSequence> {
        std::mem::take(&mut self.completed)
    }

    fn first_free_slot(&self) -> Option<usize> {
        self.active.iter().position(|a| a.is_none())
    }
}

impl Model {
    /// Run one continuous-batching iteration (see [`ContinuousBatch`]).
    /// Returns the number of requests newly scheduled into slots this step.
    pub fn continuous_step(
        &mut self,
        batch: &mut ContinuousBatch,
        sampler: &Sampler,
        gpu: Option<&GpuContext>,
    ) -> usize {
        // Size the cache on first use (rebuilding mid-run is never triggered
        // because capacity is fixed for the lifetime of the batch).
        if self.cache.as_ref().is_none_or(|c| c.batch < batch.capacity) {
            self.reserve_cache_batch(batch.capacity);
        }

        let scheduled = self.schedule_pending(batch, gpu);
        let vocab = self.config.vocab_size;

        // Sample one token per active slot.
        let mut slots: Vec<usize> = Vec::new();
        let mut tokens: Vec<u32> = Vec::new();
        let mut finish: Vec<bool> = Vec::new();
        for (b, active) in batch.active.iter_mut().enumerate() {
            let Some(seq) = active else { continue };
            let token = sampler.sample(&seq.logits);
            seq.output.push(token);
            seq.generated += 1;
            slots.push(b);
            tokens.push(token);
            finish.push(seq.generated >= seq.max_new_tokens || Some(token) == seq.eos);
        }

        if !slots.is_empty() {
            // Decode only the slots that are still running.
            let mut decode_slots: Vec<usize> = Vec::new();
            let mut decode_tokens: Vec<u32> = Vec::new();
            for (i, &b) in slots.iter().enumerate() {
                if !finish[i] {
                    decode_slots.push(b);
                    decode_tokens.push(tokens[i]);
                }
            }

            if !decode_slots.is_empty() {
                let positions: Vec<usize> = decode_slots
                    .iter()
                    .map(|&b| self.cache.as_ref().map_or(0, |c| c.seq_len(b)))
                    .collect();
                let logits = self.forward_batch_decode(&decode_tokens, &positions, gpu);
                let slice = logits.as_f32_slice();
                for (i, &b) in decode_slots.iter().enumerate() {
                    let start = i * vocab;
                    batch.active[b].as_mut().unwrap().logits
                        .copy_from_slice(&slice[start..start + vocab]);
                }
            }

            // Free finished slots and record outputs.
            for (i, done) in finish.into_iter().enumerate() {
                if !done {
                    continue;
                }
                let b = slots[i];
                let seq = batch.active[b].take().unwrap();
                batch.completed.push(CompletedSequence {
                    id: seq.id,
                    tokens: seq.output,
                });
                if let Some(cache) = self.cache.as_mut() {
                    cache.clear_slot(b);
                }
            }
        }

        batch.iterations += 1;
        scheduled
    }

    /// Run the batch to completion (all queued requests scheduled, all active
    /// sequences finished), returning completed sequences in order. Guards
    /// against infinite loops via `max_iterations`.
    pub fn continuous_run(
        &mut self,
        batch: &mut ContinuousBatch,
        sampler: &Sampler,
        gpu: Option<&GpuContext>,
        max_iterations: usize,
    ) -> Vec<CompletedSequence> {
        self.reserve_cache_batch(batch.capacity);
        let mut out = Vec::new();
        let mut iters = 0;
        while batch.is_running() && iters < max_iterations {
            self.continuous_step(batch, sampler, gpu);
            out.extend(batch.pop_completed());
            iters += 1;
        }
        out
    }

    /// Schedule queued requests into free slots, prefilling each prompt and
    /// storing the last-token logits for the first sample.
    fn schedule_pending(
        &mut self,
        batch: &mut ContinuousBatch,
        gpu: Option<&GpuContext>,
    ) -> usize {
        let mut scheduled = 0;
        loop {
            let slot = batch.first_free_slot();
            let Some(b) = slot else { break };
            let Some(req) = batch.queue.pop_front() else { break };

            if req.max_new_tokens == 0 {
                batch.completed.push(CompletedSequence {
                    id: req.id,
                    tokens: Vec::new(),
                });
                continue;
            }

            let prompt: &[u32] = if req.prompt.is_empty() { &[0] } else { &req.prompt };
            let logits = self.forward_slot(prompt, b, gpu);
            let vocab = self.config.vocab_size;
            let last_row = logits.shape()[0] - 1;
            let slice = logits.as_f32_slice();
            batch.active[b] = Some(ActiveSeq {
                id: req.id,
                max_new_tokens: req.max_new_tokens,
                eos: req.eos,
                generated: 0,
                output: Vec::new(),
                logits: slice[last_row * vocab..(last_row + 1) * vocab].to_vec(),
            });
            scheduled += 1;
        }
        scheduled
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ModelConfig;

    #[test]
    fn test_continuous_matches_generate_batch() {
        // Same prompts, same greedy sampling: continuous batching must
        // reproduce the lockstep batch output token-for-token.
        let config = ModelConfig::tiny_test();
        let sampler = Sampler::greedy();

        let mut batched = Model::new(config.clone());
        let prompts: Vec<Vec<u32>> = vec![vec![0, 1], vec![2, 3, 4], vec![5, 0, 1, 2]];
        let refs: Vec<&[u32]> = prompts.iter().map(|v| v.as_slice()).collect();
        let expected = batched.generate_batch(&refs, 6, &sampler, None);

        let mut model = Model::new(config);
        let mut batch = ContinuousBatch::new(3);
        for p in &prompts {
            batch.submit(p, 6, None);
        }
        let got = model.continuous_run(&mut batch, &sampler, None, 1000);

        assert_eq!(got.len(), 3);
        for (i, seq) in got.iter().enumerate() {
            assert_eq!(seq.tokens, expected[i], "sequence {} diverged", i);
        }
    }

    #[test]
    fn test_continuous_staggered_finish_reuses_slots() {
        // capacity 1: requests must run back-to-back, each finishing frees the
        // slot for the next. Final cache seq_len proves slot reuse.
        let config = ModelConfig::tiny_test();
        let mut model = Model::new(config.clone());
        let sampler = Sampler::greedy();

        let mut batch = ContinuousBatch::new(1);
        batch.submit(&[0u32, 1], 2, None);
        batch.submit(&[2u32, 3], 3, None);
        batch.submit(&[4u32], 1, None);

        let got = model.continuous_run(&mut batch, &sampler, None, 1000);

        assert_eq!(got.len(), 3);
        assert_eq!(got[0].tokens.len(), 2);
        assert_eq!(got[1].tokens.len(), 3);
        assert_eq!(got[2].tokens.len(), 1);
        assert!(batch.active_count() == 0);

        // With capacity 1 each sequence needs exactly its max_new_tokens
        // iterations, back to back (slot reuse, no idle steps).
        assert_eq!(batch.iterations(), 2 + 3 + 1);

        // The final slot was freed on completion.
        let cache = model.cache.as_ref().unwrap();
        assert_eq!(cache.seq_len(0), 0);
    }

    #[test]
    fn test_continuous_eos_stops_sequence() {
        // Zero-init model: greedy always samples token 0, so eos = Some(0)
        // stops every sequence after its first generated token and frees the
        // slot immediately.
        let config = ModelConfig::tiny_test();
        let mut model = Model::new(config);
        let sampler = Sampler::greedy();

        let mut batch = ContinuousBatch::new(2);
        batch.submit(&[0u32, 1], 8, Some(0));
        batch.submit(&[2u32, 3], 8, Some(0));

        let got = model.continuous_run(&mut batch, &sampler, None, 1000);

        assert_eq!(got.len(), 2);
        for seq in &got {
            assert_eq!(seq.tokens.len(), 1);
            assert_eq!(seq.tokens[0], 0);
        }
    }

    #[test]
    fn test_continuous_capacity_limits_active() {
        // 7 requests through capacity 2: active slots must never exceed 2 and
        // all requests must complete.
        let config = ModelConfig::tiny_test();
        let mut model = Model::new(config);
        let sampler = Sampler::greedy();

        let mut batch = ContinuousBatch::new(2);
        for i in 0..7u32 {
            batch.submit(&[i, i + 1], 3, None);
        }

        let mut max_active = 0;
        let mut iters = 0;
        while batch.is_running() && iters < 1000 {
            model.continuous_step(&mut batch, &sampler, None);
            max_active = max_active.max(batch.active_count());
            iters += 1;
        }

        assert!(max_active <= 2, "capacity exceeded: {}", max_active);
        assert_eq!(batch.pop_completed().len(), 7);
        assert_eq!(batch.pending(), 0);
    }

    #[test]
    fn test_continuous_zero_new_tokens() {
        let config = ModelConfig::tiny_test();
        let mut model = Model::new(config);
        let sampler = Sampler::greedy();

        let mut batch = ContinuousBatch::new(1);
        batch.submit(&[0u32, 1], 0, None);
        let got = model.continuous_run(&mut batch, &sampler, None, 100);
        assert_eq!(got.len(), 1);
        assert!(got[0].tokens.is_empty());
    }
}
