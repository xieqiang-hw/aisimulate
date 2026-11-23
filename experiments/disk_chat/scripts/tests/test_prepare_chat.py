"""Semantic tests: timestamps, prefix identity, conservative tails, provenance."""
import argparse
import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "prepare_chat.py"
spec = importlib.util.spec_from_file_location("prepare_chat", SCRIPT)
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


class PrefixTests(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(":memory:")
        self.ids = converter.HashIDs(self.db)

    def tearDown(self):
        self.db.close()

    def test_shared_blocks_and_divergent_histories(self):
        first = self.ids.convert([10, 20, 30], 48, 16)
        same_prefix = self.ids.convert([10, 20, 99], 48, 16)
        different_prefix = self.ids.convert([90, 20, 30], 48, 16)
        self.assertEqual(first[:2], same_prefix[:2])
        self.assertNotEqual(first[2], same_prefix[2])
        self.assertTrue(set(first).isdisjoint(different_prefix))

    def test_partial_source_block_is_private(self):
        first = self.ids.convert([10, 20], 21, 16)
        second = self.ids.convert([10, 20], 21, 16)
        self.assertEqual(first[0], second[0])
        self.assertNotEqual(first[1], second[1])

    def test_mooncake_expansion_does_not_invent_finer_prefixes(self):
        first = self.ids.convert([10, 20], 600, 512)
        same = self.ids.convert([10, 20], 600, 512)
        different = self.ids.convert([11, 20], 600, 512)
        self.assertEqual(len(first), 38)
        self.assertEqual(first[:32], same[:32])
        self.assertTrue(set(first[32:]).isdisjoint(same[32:]))
        self.assertTrue(set(first).isdisjoint(different))


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "input.jsonl"
        self.output = self.root / "result"
        self.args = argparse.Namespace(start_seconds=0, duration_seconds=None, types=None,
                                       context_limit=128, max_requests=0, target_model="test/model")

    def tearDown(self):
        self.temp.cleanup()

    def row(self, chat=0, parent=-1, timestamp=1.234, turn=1, **extra):
        return dict(chat_id=chat, parent_chat_id=parent, timestamp=timestamp, turn=turn,
                    input_length=32, output_length=16, hash_ids=[10, 20], type="text", **extra)

    def prepare(self, rows):
        self.source.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return converter.prepare("qwen-chat", self.source, self.output, self.args)

    def test_parent_after_child_and_exact_relative_milliseconds(self):
        manifest = self.prepare([self.row(1, 0, 1.444, 2), self.row()])
        requests = [json.loads(line) for line in (self.output / "requests.jsonl").read_text().splitlines()]
        sidecar = [json.loads(line) for line in (self.output / "source_requests.jsonl").read_text().splitlines()]
        self.assertEqual([row["timestamp"] for row in requests], [0, 210])
        self.assertEqual(manifest["session_count"], 1)
        self.assertEqual(sidecar[1]["session_id"], "qwen-chat:0")
        self.assertEqual(sidecar[0]["source_line"], 2)
        self.assertEqual(requests[0]["hash_ids"], requests[1]["hash_ids"])
        self.assertNotIn("session_id", requests[0])

    def test_filter_keeps_root_identity_when_parent_excluded(self):
        self.args.start_seconds = 0.1
        manifest = self.prepare([self.row(), self.row(1, 0, 1.444, 2)])
        source = json.loads((self.output / "source_requests.jsonl").read_text())
        self.assertEqual(manifest["request_count"], 1)
        self.assertEqual(source["session_id"], "qwen-chat:0")
        self.assertEqual(source["turn"], 2)

    def test_length_filters_never_truncate(self):
        long = self.row(1)
        long.update(input_length=128, output_length=1, hash_ids=list(range(8)))
        zero = self.row(2)
        zero["output_length"] = 0
        manifest = self.prepare([self.row(), long, zero])
        self.assertEqual(manifest["request_count"], 1)
        self.assertEqual(manifest["diagnostics"]["excluded_context_limit"], 1)
        self.assertEqual(manifest["diagnostics"]["excluded_zero_output"], 1)

    def test_bad_hash_capacity_fails_atomically(self):
        row = self.row()
        row["hash_ids"] = [10]
        with self.assertRaisesRegex(ValueError, "hash_ids count"):
            self.prepare([row])
        self.assertFalse(self.output.exists())

    def test_missing_parent_and_cycle_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing Qwen parent"):
            self.prepare([self.row(1, 9)])
        with self.assertRaisesRegex(ValueError, "cycle"):
            self.prepare([self.row(0, 1), self.row(1, 0)])

    def test_empty_selection_does_not_publish(self):
        self.args.types = ["image"]
        with self.assertRaisesRegex(ValueError, "no usable requests"):
            self.prepare([self.row()])
        self.assertFalse(self.output.exists())

    def test_existing_bundle_not_overwritten(self):
        self.prepare([self.row()])
        before = (self.output / "requests.jsonl").read_bytes()
        with self.assertRaisesRegex(ValueError, "already exists"):
            converter.prepare("qwen-chat", self.source, self.output, self.args)
        self.assertEqual(before, (self.output / "requests.jsonl").read_bytes())

    def test_time_window_is_half_open_and_limit_follows_sorting(self):
        self.args.duration_seconds = 0.2
        self.args.max_requests = 1
        manifest = self.prepare([self.row(2, timestamp=1.434), self.row(1, timestamp=1.334), self.row()])
        source = json.loads((self.output / "source_requests.jsonl").read_text())
        self.assertEqual(source["source_chat_id"], 0)
        self.assertEqual(manifest["diagnostics"]["excluded_time_window"], 1)
        self.assertEqual(manifest["diagnostics"]["excluded_request_limit"], 1)

    def test_duplicate_chat_id_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "UNIQUE constraint"):
            self.prepare([self.row(), self.row()])

    def test_nonfinite_timestamp_is_rejected(self):
        for value in [float("nan"), float("inf"), -0.1, True]:
            with self.assertRaisesRegex(ValueError, "timestamp"):
                self.prepare([self.row(timestamp=value)])


if __name__ == "__main__":
    unittest.main()
