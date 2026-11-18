// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Replay-owned, two-stage P/D handoff through a shared KV enclosure.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Result, ensure};

use crate::engine::HostBlockKey;
use crate::engine::g3_offload::{Direction, G3Tier, Probe, SharedG3Tier};
use crate::engine::{
    G3OffloadConfig, G3Scope, G3Stats, HandoffId, block_hashes, compute_next_sequence_hash,
};
use crate::replay::EnclosureHandoffConfig;

const PREFILL: usize = 0;
const DECODE: usize = 1;

struct DecodeOutput {
    parent: Option<u64>,
    tail: Vec<u32>,
    sampled: Option<u32>,
}

struct Handoff {
    keys: Vec<HostBlockKey>,
    tail: Option<HostBlockKey>,
    decode_start: usize,
    prefetched: BTreeSet<HostBlockKey>,
    jobs: Vec<(usize, u64)>,
    source_ready: bool,
    reserved: bool,
}

pub(super) struct EnclosureHandoffs {
    tier: SharedG3Tier,
    block_size: usize,
    capacity: usize,
    requests: BTreeMap<HandoffId, Handoff>,
    jobs: BTreeMap<u64, HandoffId>,
    waiting: BTreeSet<HandoffId>,
    decode_outputs: BTreeMap<HandoffId, DecodeOutput>,
    writeback: BTreeSet<HostBlockKey>,
}

impl EnclosureHandoffs {
    pub(super) fn new(
        config: &EnclosureHandoffConfig,
        block_size: usize,
        block_bytes: usize,
    ) -> Result<Self> {
        ensure!(block_size > 0, "enclosure requires positive KV block size");
        ensure!(
            config.read_bandwidth_gbps > 0.0 && config.write_bandwidth_gbps > 0.0,
            "enclosure read and write bandwidths must be positive"
        );
        ensure!(
            config
                .prefill_write_bandwidth_gbps
                .is_none_or(|rate| rate > 0.0)
                && config
                    .decode_read_bandwidth_gbps
                    .is_none_or(|rate| rate > 0.0),
            "enclosure endpoint bandwidths must be positive"
        );
        // Each endpoint only uses its respective direction in this stage.
        let tier = G3Tier::new_enclosure(
            G3OffloadConfig {
                scope: G3Scope::ClusterShared,
                num_g3_blocks: config.num_blocks,
                latency_to_first_byte_ms: config.latency_to_first_byte_ms,
                read_bandwidth_gbps: config.decode_read_bandwidth_gbps.unwrap_or(0.0),
                write_bandwidth_gbps: config.prefill_write_bandwidth_gbps.unwrap_or(0.0),
                shared_read_bandwidth_gbps: config.read_bandwidth_gbps,
                shared_write_bandwidth_gbps: config.write_bandwidth_gbps,
            },
            2,
            block_bytes,
            config.combined_bandwidth_gbps,
        )?;
        Ok(Self {
            tier,
            block_size,
            capacity: config.num_blocks,
            requests: BTreeMap::new(),
            jobs: BTreeMap::new(),
            waiting: BTreeSet::new(),
            decode_outputs: BTreeMap::new(),
            writeback: BTreeSet::new(),
        })
    }

    pub(super) fn tier(&self) -> SharedG3Tier {
        self.tier.clone()
    }

    #[cfg(test)]
    pub(super) fn register(&mut self, id: HandoffId, tokens: &[u32]) -> Result<()> {
        self.register_prompt(id, tokens, false)
    }

    pub(super) fn register_prompt(
        &mut self,
        id: HandoffId,
        tokens: &[u32],
        exact: bool,
    ) -> Result<()> {
        let blocks = tokens.len().div_ceil(self.block_size);
        ensure!(
            blocks <= self.capacity,
            "request needs {blocks} enclosure blocks but the pool holds {}",
            self.capacity
        );
        let mut parent = None;
        let mut keys = block_hashes(tokens, self.block_size)
            .map(|local| {
                let hash = parent.map_or(local, |parent| compute_next_sequence_hash(parent, local));
                parent = Some(hash);
                HostBlockKey::new(hash)
            })
            .collect::<Vec<_>>();
        // A partial block is private to this handoff. Use a domain-specific hash
        // seed for it rather than publishing it as a reusable prefix.
        let tail = (!tokens.len().is_multiple_of(self.block_size)).then(|| {
            HostBlockKey::new(xxhash_rust::xxh3::xxh3_64_with_seed(
                id.get().as_bytes(),
                0x5441494c,
            ))
        });
        keys.extend(tail);
        if exact {
            self.decode_outputs.insert(
                id,
                DecodeOutput {
                    parent,
                    tail: tokens[tokens.len() / self.block_size * self.block_size..].to_vec(),
                    sampled: None,
                },
            );
        }
        self.requests.insert(
            id,
            Handoff {
                decode_start: keys.len(),
                keys,
                tail,
                prefetched: BTreeSet::new(),
                jobs: Vec::new(),
                source_ready: false,
                reserved: false,
            },
        );
        Ok(())
    }

    pub(super) fn prefetch(&mut self, id: HandoffId, missing_tokens: usize, now: f64) {
        let Some(request) = self.requests.get_mut(&id) else {
            return;
        };
        let missing_blocks = missing_tokens.div_ceil(self.block_size);
        request.decode_start = request
            .keys
            .len()
            .checked_sub(missing_blocks)
            .expect("D reservation exceeds enclosure manifest");
        let mut tier = self.tier.lock().unwrap();
        tier.advance(now);
        let keys = request.keys[request.decode_start..]
            .iter()
            .copied()
            .filter(|key| tier.probe(DECODE, *key) == Probe::Resident)
            .collect::<Vec<_>>();
        if let Some(job) = tier.submit(DECODE, Direction::Read, &keys, now) {
            request.prefetched.extend(keys);
            request.jobs.push((DECODE, job));
            self.jobs.insert(job, id);
        }
    }

    pub(super) fn source_ready(&mut self, id: HandoffId) {
        self.requests
            .get_mut(&id)
            .expect("registered handoff")
            .source_ready = true;
        self.waiting.insert(id);
    }

    /// A sampled token obtains KV in the forward that emits its successor.
    pub(super) fn decode_output(&mut self, id: HandoffId, token: Option<u32>, finished: bool) {
        if let Some(output) = self.decode_outputs.get_mut(&id)
            && let Some(token) = token
        {
            if let Some(computed) = output.sampled.replace(token) {
                output.tail.push(computed);
                if output.tail.len() == self.block_size {
                    let local = block_hashes(&output.tail, self.block_size).next().unwrap();
                    let key = output
                        .parent
                        .map_or(local, |parent| compute_next_sequence_hash(parent, local));
                    output.parent = Some(key);
                    output.tail.clear();
                    self.writeback.insert(HostBlockKey::new(key));
                }
            }
        }
        if finished {
            self.decode_outputs.remove(&id);
        }
    }

    pub(super) fn next_event(&self) -> Option<f64> {
        let tier = self.tier.lock().unwrap();
        [
            tier.next_direction_deadline(PREFILL, Direction::Write),
            tier.next_deadline(DECODE),
        ]
        .into_iter()
        .flatten()
        .min_by(f64::total_cmp)
    }

    pub(super) fn advance(&mut self, now: f64) -> Vec<HandoffId> {
        let mut finished = Vec::new();
        loop {
            let settled = self.advance_once(now);
            let retry = !settled.is_empty() && !self.waiting.is_empty();
            finished.extend(settled);
            if !retry && self.next_event().is_none_or(|deadline| deadline > now) {
                return finished;
            }
        }
    }

    fn advance_once(&mut self, now: f64) -> Vec<HandoffId> {
        let mut tier = self.tier.lock().unwrap();
        let mut candidates = std::mem::take(&mut self.waiting);
        for worker in [PREFILL, DECODE] {
            let direction = (worker == PREFILL).then_some(Direction::Write);
            for completion in tier.take_completions(worker, now, direction) {
                let Some(id) = self.jobs.remove(&completion.id) else {
                    continue;
                };
                let request = self.requests.get_mut(&id).unwrap();
                request.jobs.retain(|&(_, job)| job != completion.id);
                if request.source_ready {
                    candidates.insert(id);
                }
            }
        }
        let mut finished = Vec::new();
        for id in candidates {
            let request = self.requests.get_mut(&id).unwrap();
            if !request.reserved {
                let Some(write) = tier.reserve_handoff(PREFILL, &request.keys, now) else {
                    self.waiting.insert(id);
                    continue;
                };
                request.reserved = true;
                if let Some(job) = write {
                    request.jobs.push((PREFILL, job));
                    self.jobs.insert(job, id);
                    continue;
                }
            }
            if request.jobs.iter().any(|&(worker, _)| worker == PREFILL) {
                continue;
            }
            let keys = request.keys[request.decode_start..]
                .iter()
                .copied()
                .filter(|key| !request.prefetched.contains(key))
                .collect::<Vec<_>>();
            if !keys.is_empty() {
                let job = tier
                    .submit(DECODE, Direction::Read, &keys, now)
                    .expect("pinned enclosure handoff blocks must be readable");
                request.prefetched.extend(keys);
                request.jobs.push((DECODE, job));
                self.jobs.insert(job, id);
            }
            if request.jobs.is_empty() {
                finished.push(id);
            }
        }
        for id in &finished {
            let request = self.requests.remove(id).unwrap();
            tier.release_handoff(PREFILL, &request.keys);
            if let Some(tail) = request.tail {
                tier.remove_private_tail(PREFILL, tail);
            }
        }
        self.writeback.retain(|key| !tier.contains(DECODE, *key));
        if self.waiting.is_empty() && !self.writeback.is_empty() {
            let keys: Vec<_> = self.writeback.iter().copied().collect();
            if let Some(job) = tier.submit(DECODE, Direction::Write, &keys, now) {
                for key in tier.job_keys(job) {
                    self.writeback.remove(key);
                }
            }
        }
        finished
    }

    pub(super) fn has_writeback(&self) -> bool {
        !self.writeback.is_empty() || self.tier.lock().unwrap().pending_io_counts(DECODE).1 > 0
    }

    pub(super) fn cancel(&mut self, id: HandoffId, now: f64) {
        self.waiting.remove(&id);
        self.decode_outputs.remove(&id);
        let Some(request) = self.requests.remove(&id) else {
            return;
        };
        let mut tier = self.tier.lock().unwrap();
        if request.reserved {
            tier.release_handoff(PREFILL, &request.keys);
        }
        for (_, job) in request.jobs {
            self.jobs.remove(&job);
            tier.cancel(job, now);
        }
        if let Some(tail) = request.tail {
            tier.remove_private_tail(PREFILL, tail);
        }
    }

    pub(super) fn stats(&self) -> G3Stats {
        self.tier.lock().unwrap().snapshot()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn capacity_waiter_can_retry_after_an_inflight_handoff_is_canceled() {
        let mut enclosure = EnclosureHandoffs::new(
            &EnclosureHandoffConfig {
                num_blocks: 2,
                read_bandwidth_gbps: 1.0,
                write_bandwidth_gbps: 1.0,
                combined_bandwidth_gbps: Some(1.0),
                prefill_write_bandwidth_gbps: None,
                decode_read_bandwidth_gbps: None,
                latency_to_first_byte_ms: 1.0,
            },
            64,
            1024,
        )
        .unwrap();
        let first = HandoffId::new(Uuid::from_u128(1));
        let second = HandoffId::new(Uuid::from_u128(2));
        for (id, token) in [(first, 1), (second, 2)] {
            enclosure.register(id, &[token; 65]).unwrap();
            enclosure.prefetch(id, 128, 0.0);
            enclosure.source_ready(id);
        }
        assert!(enclosure.advance(0.0).is_empty());
        assert_eq!(enclosure.stats().pending_blocks, 2);
        assert_eq!(enclosure.stats().write.submitted_jobs, 1);
        enclosure.cancel(first, 0.5);
        assert!(enclosure.advance(0.5).is_empty());
        assert_eq!(enclosure.stats().write.cancelled_jobs, 1);
        assert_eq!(enclosure.stats().write.submitted_jobs, 2);
        assert!(enclosure.advance(2.0).is_empty());
        assert_eq!(enclosure.advance(4.0), [second]);
        assert_eq!(enclosure.stats().resident_blocks, 1);
        assert_eq!(enclosure.stats().pending_blocks, 0);
    }
}
