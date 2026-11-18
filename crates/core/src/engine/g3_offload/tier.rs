// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Offline secondary storage. Each Replay deployment owns its G3 tier.
//! No global namespace, wall clock, payload allocation, or hardware connection.

use crate::engine::config::{G3OffloadConfig, G3Scope};
use crate::engine::host_offload::HostBlockKey;
use crate::engine::offload_lru::Lru;
pub(crate) use crate::engine::offload_transfer::Direction;
use crate::engine::offload_transfer::{self, FairTransfers, Link};
use anyhow::{Result, ensure};
use rustc_hash::{FxHashMap, FxHashSet};
use serde::Serialize;
use std::collections::BTreeMap;
use std::num::NonZeroU32;
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug, Default, Serialize, PartialEq)]
pub struct G3IoStats {
    pub submitted_jobs: u64,
    pub completed_jobs: u64,
    pub cancelled_jobs: u64,
    pub completed_bytes: u64,
    pub transfer_ms: f64,
}

#[derive(Clone, Debug, Default, Serialize, PartialEq)]
pub struct G3Stats {
    /// Every actual block probe, including retries; pending is not a hit.
    pub lookup_probes: u64,
    pub lookup_hits: u64,
    pub lookup_pending: u64,
    pub read: G3IoStats,
    pub write: G3IoStats,
    pub evictions: u64,
    pub cross_worker_read_blocks: u64,
    /// Complete generated blocks written by Decode to the enclosure.
    #[serde(skip_serializing_if = "is_zero")]
    pub decode_writeback_blocks: u64,
    /// P-side direct reads of complete blocks written by Decode.
    #[serde(skip_serializing_if = "is_zero")]
    pub prefill_decode_reuse_blocks: u64,
    /// Completed P HBM-to-DRAM promotions, separate from enclosure writes.
    #[serde(skip_serializing_if = "is_zero")]
    pub prefill_dram_promotion_blocks: u64,
    #[serde(skip_serializing_if = "is_zero")]
    pub prefill_dram_promotion_bytes: u64,
    pub resident_blocks: usize,
    pub pending_blocks: usize,
    /// Times a request stopped restoring from G3 after too many restore
    /// rounds while its DP rank made no progress. Omitted when zero.
    #[serde(skip_serializing_if = "is_zero")]
    pub bypassed_restores: u64,
}

pub(crate) fn is_zero(value: &u64) -> bool {
    *value == 0
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Probe {
    Miss,
    Pending,
    Resident,
}

#[derive(Clone, Copy)]
struct Entry {
    pending_job: Option<u64>,
    pins: usize,
    writer: usize,
    /// This block's node in its pool's LRU once its write completes.
    lru_slot: Option<NonZeroU32>,
}

/// Blocks and recency of one G3 namespace.
#[derive(Default)]
struct Pool {
    /// Resident and pending blocks; their count is the pool's used capacity.
    entries: FxHashMap<HostBlockKey, Entry>,
    /// Resident blocks, least recently probed or transferred first. A
    /// completed job's blocks become newest together, in key order.
    lru: Lru,
}

struct Job {
    worker: usize,
    direction: Direction,
    keys: Vec<HostBlockKey>,
    submitted: f64,
    observed_end: crate::engine::offload_transfer::TransferEnd,
}

#[derive(Clone, Debug)]
pub(crate) struct Completion {
    pub id: u64,
    pub direction: Direction,
    pub keys: Vec<HostBlockKey>,
    pub at_ms: f64,
}

pub(crate) type SharedG3Tier = Arc<Mutex<G3Tier>>;

pub(crate) struct G3Tier {
    config: G3OffloadConfig,
    next_worker_id: usize,
    block_bytes: usize,
    combined_bandwidth_gbps: Option<f64>,
    enclosure: bool,
    pools: BTreeMap<usize, Pool>,
    jobs: BTreeMap<u64, Job>,
    transfers: FairTransfers,
    completed: BTreeMap<usize, Vec<Completion>>,
    last_job_id: u64,
    pub(crate) completion_epoch: u64,
    pub stats: G3Stats,
}

impl G3Tier {
    pub(crate) fn new(
        config: G3OffloadConfig,
        workers: usize,
        block_bytes: usize,
    ) -> Result<SharedG3Tier> {
        Self::new_with_bandwidth(config, workers, block_bytes, None, false)
    }

    /// Shared enclosure endpoint for direct HBM I/O. Worker IDs identify
    /// compute endpoints and block_bytes is the payload across all TP shards.
    /// A combined budget limits both directions; None uses separate limits.
    pub(crate) fn new_enclosure(
        config: G3OffloadConfig,
        workers: usize,
        block_bytes: usize,
        combined_bandwidth_gbps: Option<f64>,
    ) -> Result<SharedG3Tier> {
        ensure!(
            config.scope == G3Scope::ClusterShared,
            "enclosure requires a shared pool"
        );
        if let Some(rate) = combined_bandwidth_gbps {
            ensure!(
                rate.is_finite() && rate > 0.0,
                "combined enclosure bandwidth must be finite and positive"
            );
        }
        Self::new_with_bandwidth(config, workers, block_bytes, combined_bandwidth_gbps, true)
    }

    fn new_with_bandwidth(
        config: G3OffloadConfig,
        workers: usize,
        block_bytes: usize,
        combined_bandwidth_gbps: Option<f64>,
        enclosure: bool,
    ) -> Result<SharedG3Tier> {
        config.validate()?;
        ensure!(block_bytes > 0, "G3 requires positive block bytes");
        let bytes = config
            .num_g3_blocks
            .checked_mul(block_bytes)
            .ok_or_else(|| anyhow::anyhow!("G3 capacity bytes overflow"))?;
        for bandwidth in [config.read_bandwidth_gbps, config.write_bandwidth_gbps]
            .into_iter()
            .chain(
                (config.scope == G3Scope::ClusterShared)
                    .then_some([
                        config.shared_read_bandwidth_gbps,
                        config.shared_write_bandwidth_gbps,
                    ])
                    .into_iter()
                    .flatten(),
            )
            .chain(combined_bandwidth_gbps)
        {
            ensure!(
                bandwidth == 0.0 || (bytes as f64 / (bandwidth * 1e6)).is_finite(),
                "G3 transfer duration overflow"
            );
        }
        Ok(Arc::new(Mutex::new(Self {
            config,
            next_worker_id: workers,
            block_bytes,
            combined_bandwidth_gbps,
            enclosure,
            pools: BTreeMap::new(),
            jobs: BTreeMap::new(),
            transfers: FairTransfers::with_combined_pool_directions(
                combined_bandwidth_gbps.is_some(),
            ),
            completed: (0..workers).map(|worker| (worker, Vec::new())).collect(),
            last_job_id: 0,
            completion_epoch: 0,
            stats: G3Stats::default(),
        })))
    }

    pub(crate) fn current_time_ms(&self) -> f64 {
        self.transfers.now()
    }

    pub(crate) fn register_worker(&mut self, worker: usize) -> Result<()> {
        ensure!(
            worker >= self.next_worker_id,
            "G3 worker IDs cannot be reused"
        );
        self.next_worker_id = worker
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("G3 worker ID overflow"))?;
        self.completed.insert(worker, Vec::new());
        Ok(())
    }

    pub(crate) fn has_work(&self, worker: usize) -> bool {
        !self.completed[&worker].is_empty() || self.transfers.has_link_jobs(worker as u64)
    }

    pub(crate) fn unregister_worker(&mut self, worker: usize) -> Result<()> {
        ensure!(
            self.completed.contains_key(&worker),
            "G3 worker is not registered"
        );
        ensure!(
            !self.has_work(worker),
            "cannot unregister G3 worker with pending I/O"
        );
        self.completed.remove(&worker);
        if self.config.scope == G3Scope::WorkerLocal {
            self.pools.remove(&worker);
        }
        Ok(())
    }

    fn pool(&self, worker: usize) -> usize {
        assert!(
            self.completed.contains_key(&worker),
            "G3 worker is not registered"
        );
        match self.config.scope {
            G3Scope::WorkerLocal => worker,
            G3Scope::ClusterShared => 0,
        }
    }

    /// Read-only availability for a capacity check; does not touch LRU or stats.
    pub(crate) fn contains(&self, worker: usize, key: HostBlockKey) -> bool {
        self.pools
            .get(&self.pool(worker))
            .is_some_and(|pool| pool.entries.contains_key(&key))
    }

    pub(crate) fn probe(&mut self, worker: usize, key: HostBlockKey) -> Probe {
        let pool = self.pool(worker);
        self.stats.lookup_probes += 1;
        let Some(Pool { entries, lru }) = self.pools.get_mut(&pool) else {
            return Probe::Miss;
        };
        match entries.get_mut(&key) {
            None => Probe::Miss,
            Some(entry) if entry.pending_job.is_some() => {
                self.stats.lookup_pending += 1;
                Probe::Pending
            }
            Some(entry) => {
                lru.touch(&mut entry.lru_slot, key);
                self.stats.lookup_hits += 1;
                Probe::Resident
            }
        }
    }

    pub(crate) fn submit(
        &mut self,
        worker: usize,
        direction: Direction,
        keys: &[HostBlockKey],
        now: f64,
    ) -> Option<u64> {
        self.submit_io(worker, direction, keys, now, now, false)
    }

    pub(crate) fn submit_after(
        &mut self,
        worker: usize,
        direction: Direction,
        keys: &[HostBlockKey],
        now: f64,
        not_before: f64,
    ) -> Option<u64> {
        self.submit_io(worker, direction, keys, now, not_before, false)
    }

    /// Check and reserve the full handoff in one transaction. Pending blocks
    /// are retried after their owner finishes, without taking partial holds.
    pub(crate) fn reserve_handoff(
        &mut self,
        worker: usize,
        keys: &[HostBlockKey],
        now: f64,
    ) -> Option<Option<u64>> {
        self.advance(now);
        let pool = self.pool(worker);
        if keys.iter().any(|key| {
            self.pools
                .get(&pool)
                .and_then(|pool| pool.entries.get(key))
                .is_some_and(|entry| entry.pending_job.is_some())
        }) {
            return None;
        }
        let missing = keys.iter().any(|key| !self.contains(worker, *key));
        let job = if missing {
            Some(self.submit_io(worker, Direction::Write, keys, now, now, true)?)
        } else {
            None
        };
        if let Some(pool) = self.pools.get_mut(&pool) {
            for key in keys {
                pool.entries.get_mut(key).unwrap().pins += 1;
            }
        }
        Some(job)
    }

    pub(crate) fn release_handoff(&mut self, worker: usize, keys: &[HostBlockKey]) {
        let pool = self.pool(worker);
        if let Some(pool) = self.pools.get_mut(&pool) {
            for key in keys {
                pool.entries.get_mut(key).unwrap().pins -= 1;
            }
        }
    }

    pub(crate) fn remove_private_tail(&mut self, worker: usize, key: HostBlockKey) {
        let pool = self.pool(worker);
        if let Some(pool) = self.pools.get_mut(&pool) {
            if let Some(mut entry) = pool.entries.remove(&key) {
                assert!(entry.pending_job.is_none() && entry.pins == 0);
                pool.lru.remove(&mut entry.lru_slot);
            }
        }
    }

    fn submit_io(
        &mut self,
        worker: usize,
        direction: Direction,
        keys: &[HostBlockKey],
        now: f64,
        not_before: f64,
        require_all: bool,
    ) -> Option<u64> {
        // Commands from a lagging rank can never schedule work in the past.
        self.advance(now);
        let pool = self.pool(worker);
        let protected: FxHashSet<_> = keys.iter().copied().collect();
        let Pool { entries, lru } = match direction {
            Direction::Write => self.pools.entry(pool).or_default(),
            Direction::Read => self.pools.get_mut(&pool)?,
        };
        let mut seen = FxHashSet::default();
        let mut keys = match direction {
            Direction::Write => keys
                .iter()
                .copied()
                .filter(|key| !entries.contains_key(key) && seen.insert(*key))
                .collect::<Vec<_>>(),
            Direction::Read => keys.to_vec(),
        };
        if keys.is_empty() {
            return None;
        }
        let id = self.last_job_id + 1;
        if direction == Direction::Write {
            let free = self
                .config
                .num_g3_blocks
                .checked_sub(entries.len())
                .expect("G3 pool exceeds its capacity");
            let victims = lru.victims(keys.len().saturating_sub(free), |key| {
                entries[&key].pins == 0 && !protected.contains(&key)
            });
            // Keep the leading new blocks that fit, preserving any blocks of
            // this prefix already stored. Pending writes and read pins retain
            // their capacity; only accepted blocks own source pins and bytes.
            if require_all && keys.len() > free + victims.len() {
                return None;
            }
            keys.truncate(free + victims.len());
            if keys.is_empty() {
                return None;
            }
            for key in victims {
                let mut entry = entries.remove(&key).unwrap();
                lru.remove(&mut entry.lru_slot);
                self.stats.evictions += 1;
            }
            for key in &keys {
                let entry = Entry {
                    pending_job: Some(id),
                    pins: 0,
                    writer: worker,
                    lru_slot: None,
                };
                entries.insert(*key, entry);
            }
        } else {
            if keys
                .iter()
                .any(|key| entries.get(key).is_none_or(|e| e.pending_job.is_some()))
            {
                return None;
            }
            for key in &keys {
                entries.get_mut(key).unwrap().pins += 1;
            }
        }
        self.last_job_id = id;
        self.io_mut(direction).submitted_jobs += 1;
        let now = self.transfers.now();
        let (worker_gbps, shared_gbps) = match direction {
            Direction::Read => (
                self.config.read_bandwidth_gbps,
                self.config.shared_read_bandwidth_gbps,
            ),
            Direction::Write => (
                self.config.write_bandwidth_gbps,
                self.config.shared_write_bandwidth_gbps,
            ),
        };
        self.transfers.submit(offload_transfer::Job {
            id,
            direction,
            ready: now.max(not_before) + self.config.latency_to_first_byte_ms,
            bytes: (keys.len() * self.block_bytes) as f64,
            links: [
                Link {
                    id: worker as u64,
                    rate: offload_transfer::bytes_per_ms(worker_gbps),
                },
                match self.config.scope {
                    G3Scope::WorkerLocal => Link::UNLIMITED,
                    G3Scope::ClusterShared => Link {
                        id: 0,
                        rate: offload_transfer::bytes_per_ms(
                            self.combined_bandwidth_gbps.unwrap_or(shared_gbps),
                        ),
                    },
                },
            ],
        });
        self.jobs.insert(
            id,
            Job {
                worker,
                direction,
                keys,
                submitted: now,
                observed_end: Default::default(),
            },
        );
        Some(id)
    }

    pub(crate) fn observe_job(&mut self, id: u64) -> Option<Arc<std::sync::OnceLock<f64>>> {
        Some(self.jobs.get_mut(&id)?.observed_end.observe())
    }

    pub(crate) fn observe_pending_write(
        &mut self,
        worker: usize,
        key: HostBlockKey,
    ) -> Option<Arc<std::sync::OnceLock<f64>>> {
        let id = self
            .pools
            .get(&self.pool(worker))?
            .entries
            .get(&key)?
            .pending_job?;
        self.observe_job(id)
    }

    pub(crate) fn job_keys(&self, id: u64) -> &[HostBlockKey] {
        &self.jobs[&id].keys
    }

    fn io_mut(&mut self, direction: Direction) -> &mut G3IoStats {
        match direction {
            Direction::Read => &mut self.stats.read,
            Direction::Write => &mut self.stats.write,
        }
    }

    pub(crate) fn next_deadline(&self, worker: usize) -> Option<f64> {
        self.transfers
            .next_event()
            .into_iter()
            .chain(self.completed[&worker].iter().map(|done| done.at_ms))
            .min_by(f64::total_cmp)
    }

    pub(crate) fn advance(&mut self, target: f64) {
        for (id, at_ms) in self.transfers.advance(target) {
            let job = self
                .jobs
                .remove(&id)
                .expect("finished G3 job lost its metadata");
            self.finish(id, job, at_ms);
        }
    }

    fn finish(&mut self, id: u64, job: Job, now: f64) {
        job.observed_end.finish(now);
        self.completion_epoch += 1;
        let pool = self.pool(job.worker);
        let Pool { entries, lru } = self.pools.get_mut(&pool).expect("G3 pool");
        let mut cohort = job.keys.clone();
        cohort.sort_unstable();
        for key in cohort {
            let entry = entries.get_mut(&key).unwrap();
            match job.direction {
                Direction::Write => entry.pending_job = None,
                Direction::Read => {
                    entry.pins -= 1;
                    self.stats.cross_worker_read_blocks += u64::from(entry.writer != job.worker);
                    if self.enclosure && job.worker == 0 && entry.writer == 1 {
                        self.stats.prefill_decode_reuse_blocks += 1;
                    }
                }
            }
            lru.touch(&mut entry.lru_slot, key);
        }
        if self.enclosure && job.worker == 1 && job.direction == Direction::Write {
            self.stats.decode_writeback_blocks += job.keys.len() as u64;
        }
        let bytes = (job.keys.len() * self.block_bytes) as u64;
        let io = self.io_mut(job.direction);
        io.completed_jobs += 1;
        io.completed_bytes += bytes;
        io.transfer_ms += now - job.submitted;
        self.completed
            .get_mut(&job.worker)
            .unwrap()
            .push(Completion {
                id,
                direction: job.direction,
                keys: job.keys,
                at_ms: now,
            });
    }

    pub(crate) fn next_direction_deadline(
        &self,
        worker: usize,
        direction: Direction,
    ) -> Option<f64> {
        self.transfers
            .next_event()
            .into_iter()
            .chain(
                self.completed[&worker]
                    .iter()
                    .filter(|done| done.direction == direction)
                    .map(|done| done.at_ms),
            )
            .min_by(f64::total_cmp)
    }

    pub(crate) fn take_completed(&mut self, worker: usize, now: f64) -> Vec<Completion> {
        self.take_completions(worker, now, None)
    }

    pub(crate) fn take_completions(
        &mut self,
        worker: usize,
        now: f64,
        direction: Option<Direction>,
    ) -> Vec<Completion> {
        self.advance(now);
        let mut result = Vec::new();
        self.completed.get_mut(&worker).unwrap().retain(|done| {
            if done.at_ms <= now && direction.is_none_or(|direction| done.direction == direction) {
                result.push(done.clone());
                false
            } else {
                true
            }
        });
        result
    }

    pub(crate) fn cancel(&mut self, id: u64, now: f64) {
        self.advance(now);
        if self.transfers.cancel(id) {
            let job = self.jobs.remove(&id).unwrap();
            job.observed_end.finish(self.transfers.now());
            let pool = self.pool(job.worker);
            let entries = &mut self.pools.get_mut(&pool).unwrap().entries;
            for key in &job.keys {
                if job.direction == Direction::Write {
                    entries.remove(key);
                } else {
                    entries.get_mut(key).unwrap().pins -= 1;
                }
            }
            self.io_mut(job.direction).cancelled_jobs += 1;
        }
    }

    pub(crate) fn pending_io_counts(&self, worker: usize) -> (usize, usize) {
        let mut counts = (0, 0);
        for job in self.jobs.values().filter(|job| job.worker == worker) {
            match job.direction {
                Direction::Read => counts.0 += 1,
                Direction::Write => counts.1 += 1,
            }
        }
        counts
    }

    pub(crate) fn snapshot(&self) -> G3Stats {
        let mut stats = self.stats.clone();
        let entries = self.pools.values().flat_map(|pool| pool.entries.values());
        stats.pending_blocks = entries.clone().filter(|e| e.pending_job.is_some()).count();
        stats.resident_blocks = entries.count() - stats.pending_blocks;
        stats
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
