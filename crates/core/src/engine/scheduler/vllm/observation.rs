// SPDX-License-Identifier: Apache-2.0
//! Compact, bounded-memory observations; no scheduler decisions depend on them.
use std::collections::HashMap;
use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use flate2::{Compression, write::GzEncoder};
use serde::{Serialize, Serializer};
use uuid::Uuid;

use crate::engine::PressureEvent;

pub(super) const COLUMNS: [&str; 13] = [
    "stream_id",
    "iteration_id",
    "start_ms",
    "compute_start_ms",
    "end_ms",
    "prefill_bs",
    "prefill_new_tokens",
    "prefill_prefix_tokens",
    "decode_bs",
    "decode_kv_tokens_sum",
    "decode_kv_tokens_max",
    "decode_generated_tokens",
    "preemptions_this_pass",
];

#[derive(Clone, Debug, Serialize)]
pub(super) struct Identity {
    pub role: &'static str,
    pub worker_id: u64,
    pub dp_rank: u32,
    pub hbm_total_blocks: usize,
    pub hbm_grouped: bool,
    pub block_size: usize,
    pub max_sequences: Option<usize>,
    pub max_batched_tokens: Option<usize>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub(super) struct Snapshot {
    pub running_requests: usize,
    pub waiting_admission_requests: usize,
    pub connector_waiting_requests: usize,
    pub h2d_loading_requests: usize,
    pub g3_staging_requests: Option<usize>,
    pub g3_pending_read_jobs: Option<usize>,
    pub g3_pending_write_jobs: Option<usize>,
    pub pending_destination_requests: usize,
    pub reserved_destination_requests: usize,
    pub source_held_requests: usize,
    pub hbm_active_blocks: usize,
    pub hbm_inactive_blocks: usize,
    pub preemptions_total: u64,
    pub running_not_scheduled_after: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Iteration {
    pub stream_id: u64,
    pub iteration_id: u64,
    pub start_ms: f64,
    pub compute_start_ms: f64,
    pub end_ms: f64,
    pub prefill_bs: usize,
    pub prefill_new_tokens: usize,
    pub prefill_prefix_tokens: usize,
    pub decode_bs: usize,
    pub decode_kv_tokens_sum: usize,
    pub decode_kv_tokens_max: usize,
    pub decode_generated_tokens: usize,
    pub preemptions_this_pass: usize,
}

// A fixed tuple avoids Value maps, key strings, and per-iteration heap allocation.
impl Serialize for Iteration {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        (
            self.stream_id,
            self.iteration_id,
            self.start_ms,
            self.compute_start_ms,
            self.end_ms,
            self.prefill_bs,
            self.prefill_new_tokens,
            self.prefill_prefix_tokens,
            self.decode_bs,
            self.decode_kv_tokens_sum,
            self.decode_kv_tokens_max,
            self.decode_generated_tokens,
            self.preemptions_this_pass,
        )
            .serialize(serializer)
    }
}

pub(super) struct AdmissionDetail {
    pub request_id: Uuid,
    pub num_preemptions: Option<usize>,
    pub reused_input_tokens: usize,
}

pub(super) struct PassObservation {
    pub snapshot: Snapshot,
    pub iteration: Iteration,
    pub admissions: Vec<AdmissionDetail>,
}

#[derive(Serialize)]
pub(super) struct Event<'a> {
    pub schema_version: u32,
    pub stream_id: u64,
    pub event: &'static str,
    pub at_ms: f64,
    pub request_id: Uuid,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_preemptions: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reused_input_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<&'a PressureEvent>,
}

enum Stream {
    Gzip(BufWriter<GzEncoder<BufWriter<File>>>),
    Plain(BufWriter<File>),
}

impl Stream {
    fn open(path: &Path, name: &str, compression: &str) -> anyhow::Result<Self> {
        let suffix = if compression == "gzip" {
            ".jsonl.gz"
        } else {
            ".jsonl"
        };
        let file = BufWriter::with_capacity(
            262144,
            File::options()
                .write(true)
                .create_new(true)
                .open(path.join(format!("{name}{suffix}")))?,
        );
        Ok(if compression == "gzip" {
            Self::Gzip(BufWriter::with_capacity(
                262144,
                GzEncoder::new(file, Compression::fast()),
            ))
        } else {
            Self::Plain(file)
        })
    }

    fn finish(&mut self) -> std::io::Result<()> {
        match self {
            Self::Gzip(stream) => {
                stream.flush()?;
                stream.get_mut().try_finish()?;
                stream.get_mut().get_mut().flush()
            }
            Self::Plain(stream) => stream.flush(),
        }
    }
}

impl Write for Stream {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self {
            Self::Gzip(s) => s.write(buf),
            Self::Plain(s) => s.write(buf),
        }
    }
    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            Self::Gzip(s) => s.flush(),
            Self::Plain(s) => s.flush(),
        }
    }
}

type Shared = Arc<Mutex<Streams>>;

struct Streams {
    iterations: Stream,
    states: Stream,
    events: Stream,
    metadata: File,
    compression: String,
    identities: Vec<Identity>,
}

impl Streams {
    fn open(path: &Path, compression: &str) -> anyhow::Result<Self> {
        create_dir_all(path)?;
        let mut streams = Self {
            metadata: File::options()
                .write(true)
                .create_new(true)
                .open(path.join("scheduler_trace_metadata.json"))?,
            iterations: Stream::open(path, "iteration_metrics", compression)?,
            states: Stream::open(path, "scheduler_states", compression)?,
            events: Stream::open(path, "scheduling_events", compression)?,
            compression: compression.into(),
            identities: Vec::new(),
        };
        streams.write_metadata()?;
        Ok(streams)
    }

    fn write_metadata(&mut self) -> anyhow::Result<()> {
        self.metadata.seek(SeekFrom::Start(0))?;
        self.metadata.set_len(0)?;
        serde_json::to_writer(
            &mut self.metadata,
            &serde_json::json!({
                "schema_version": 2, "iteration_columns": COLUMNS,
                "streams": self.identities, "compression": self.compression,
                "state_columns": ["stream_id", "iteration_id", "at_ms", "state"],
                "gauge_boundary": "after_schedule_before_emission",
                "state_change_policy": "first_snapshot_and_every_change",
            }),
        )?;
        self.metadata.flush()?;
        Ok(())
    }
}

impl Drop for Streams {
    fn drop(&mut self) {
        for stream in [&mut self.iterations, &mut self.states, &mut self.events] {
            if let Err(error) = stream.finish() {
                eprintln!("scheduler observation finalization failed: {error}");
            }
        }
    }
}

pub(super) struct Recorder {
    shared: Shared,
    stream_id: Option<u64>,
    last_state: Option<Snapshot>,
}

impl Recorder {
    pub(super) fn from_env() -> anyhow::Result<Option<Self>> {
        let Some(path) = std::env::var_os("AISIM_SCHEDULER_TRACE_DIR") else {
            return Ok(None);
        };
        let compression =
            std::env::var("AISIM_SCHEDULER_TRACE_COMPRESSION").unwrap_or_else(|_| "gzip".into());
        Ok(Some(Self::open_with_compression(
            Path::new(&path),
            &compression,
        )?))
    }

    #[cfg(test)]
    pub(super) fn open(path: &Path) -> anyhow::Result<Self> {
        Self::open_with_compression(path, "gzip")
    }

    fn open_with_compression(path: &Path, compression: &str) -> anyhow::Result<Self> {
        anyhow::ensure!(
            matches!(compression, "gzip" | "none"),
            "trace compression must be gzip or none"
        );
        static OPEN: OnceLock<Mutex<HashMap<PathBuf, Weak<Mutex<Streams>>>>> = OnceLock::new();
        let mut opened = OPEN
            .get_or_init(Default::default)
            .lock()
            .map_err(|_| anyhow::anyhow!("scheduler observation registry poisoned"))?;
        let shared = match opened.get(path).and_then(Weak::upgrade) {
            Some(stream) => {
                anyhow::ensure!(
                    stream.lock().unwrap().compression == compression,
                    "trace compression changed within a run"
                );
                stream
            }
            None => {
                let stream = Arc::new(Mutex::new(Streams::open(path, compression)?));
                opened.insert(path.to_owned(), Arc::downgrade(&stream));
                stream
            }
        };
        Ok(Self {
            shared,
            stream_id: None,
            last_state: None,
        })
    }

    pub(super) fn record_pass(
        &mut self,
        identity: Identity,
        mut row: Iteration,
        state: Snapshot,
    ) -> anyhow::Result<u64> {
        let mut streams = self
            .shared
            .lock()
            .map_err(|_| anyhow::anyhow!("scheduler observation stream poisoned"))?;
        let id = match self.stream_id {
            Some(id) => id,
            None => {
                let id = streams.identities.len() as u64;
                streams.identities.push(identity);
                streams.write_metadata()?;
                self.stream_id = Some(id);
                id
            }
        };
        row.stream_id = id;
        if self.last_state != Some(state) {
            serde_json::to_writer(
                &mut streams.states,
                &(id, row.iteration_id, row.start_ms, state),
            )?;
            streams.states.write_all(b"\n")?;
            self.last_state = Some(state);
        }
        serde_json::to_writer(&mut streams.iterations, &row)?;
        streams.iterations.write_all(b"\n")?;
        Ok(id)
    }

    pub(super) fn event(&self, value: &Event<'_>) -> anyhow::Result<()> {
        let mut streams = self
            .shared
            .lock()
            .map_err(|_| anyhow::anyhow!("scheduler observation stream poisoned"))?;
        serde_json::to_writer(&mut streams.events, value)?;
        streams.events.write_all(b"\n")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    fn identity() -> Identity {
        Identity {
            role: "prefill",
            worker_id: 0,
            dp_rank: 0,
            hbm_total_blocks: 32,
            hbm_grouped: false,
            block_size: 4,
            max_sequences: Some(2),
            max_batched_tokens: Some(8),
        }
    }
    fn read_rows(path: &Path) -> Vec<serde_json::Value> {
        let mut text = String::new();
        flate2::read::GzDecoder::new(File::open(path).unwrap())
            .read_to_string(&mut text)
            .unwrap();
        text.lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    #[test]
    fn shared_streams_finish_without_retaining_rows() {
        let dir = tempfile::tempdir().unwrap();
        let mut a = Recorder::open(dir.path()).unwrap();
        let mut b = Recorder::open(dir.path()).unwrap();
        a.record_pass(
            identity(),
            Iteration {
                prefill_bs: 2,
                ..Default::default()
            },
            Snapshot::default(),
        )
        .unwrap();
        a.record_pass(
            identity(),
            Iteration {
                prefill_bs: 3,
                ..Default::default()
            },
            Snapshot::default(),
        )
        .unwrap();
        let mut changed = Snapshot::default();
        changed.hbm_active_blocks = 1;
        a.record_pass(identity(), Iteration::default(), changed)
            .unwrap();
        b.record_pass(identity(), Iteration::default(), Snapshot::default())
            .unwrap();
        drop(a);
        drop(b);
        let rows = read_rows(&dir.path().join("iteration_metrics.jsonl.gz"));
        assert_eq!(rows.len(), 4);
        assert_eq!(rows[0][5], 2);
        assert_eq!(rows[1][5], 3);
        let states = read_rows(&dir.path().join("scheduler_states.jsonl.gz"));
        assert_eq!(
            states.len(),
            3,
            "unchanged state is not repeated; each stream gets a baseline"
        );
        let meta: serde_json::Value = serde_json::from_reader(
            File::open(dir.path().join("scheduler_trace_metadata.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(meta["streams"].as_array().unwrap().len(), 2);
        assert!(
            Recorder::open(dir.path()).is_err(),
            "never append to a prior run"
        );
    }

    #[test]
    fn uncompressed_streams_keep_the_same_schema() {
        let dir = tempfile::tempdir().unwrap();
        let mut recorder = Recorder::open_with_compression(dir.path(), "none").unwrap();
        recorder
            .record_pass(identity(), Iteration::default(), Snapshot::default())
            .unwrap();
        drop(recorder);
        let rows = std::fs::read_to_string(dir.path().join("iteration_metrics.jsonl")).unwrap();
        assert_eq!(
            serde_json::from_str::<Vec<f64>>(rows.trim()).unwrap().len(),
            COLUMNS.len()
        );
    }
}
