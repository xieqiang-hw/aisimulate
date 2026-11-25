"""Compact enclosure overlay preservation and formal runner selection."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load("prepare_enclosure_overlay")
runner = load("run_enclosure_sample")


class EnclosureWorkflowTest(unittest.TestCase):
    def make_bundle(self, root):
        bundle = root / "compact"
        bundle.mkdir()
        rows = []
        sidecar = []
        for i, (timestamp, session) in enumerate([(0, "a"), (30, "b"), (110, "a"), (130, "b")]):
            rows.append({"request_id": str(i), "input_length": 100, "output_length": 70,
                         "hash_ids": list(range(11)), "hash_ids_scope": "input_output",
                         "hash_block_size": 16, "timestamp": timestamp})
            sidecar.append({"trace_row": i, "session_id": session, "timestamp": timestamp})
        files = {}
        for name, records in [("requests.jsonl", rows), ("source_requests.jsonl", sidecar)]:
            data = "".join(json.dumps(row) + "\n" for row in records).encode()
            (bundle / name).write_bytes(data)
            files[name] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        manifest = {"format": "mooncake", "request_count": 4, "trace_block_size": 16,
                    "hash_block_size": 16, "hash_ids_scope": "input_output",
                    "includes_input_token_ids": False, "includes_output_token_ids": False,
                    "files": files}
        (bundle / "manifest.json").write_text(json.dumps(manifest))
        return bundle, rows

    def test_overlay_retains_complete_hashes_and_session_intervals(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle, original = self.make_bundle(root)
            prepare.main(["--bundle", str(bundle), "--factors", "8"])
            target = root / "compact-overlay-8x"
            manifest = runner.validate_compact_bundle(target)
            self.assertEqual(manifest["overlay"]["factor"], 8)
            actual = [json.loads(line) for line in (target / "requests.jsonl").read_text().splitlines()]
            original_by_id = {row["request_id"]: row for row in original}
            actual_by_id = {row["request_id"]: row for row in actual}
            for row in actual:
                expected = original_by_id[row["request_id"]]
                self.assertEqual({k: v for k, v in row.items() if k != "timestamp"},
                                 {k: v for k, v in expected.items() if k != "timestamp"})
            self.assertEqual(actual_by_id["2"]["timestamp"] - actual_by_id["0"]["timestamp"], 110)
            self.assertEqual(actual_by_id["3"]["timestamp"] - actual_by_id["1"]["timestamp"], 100)

    def test_runner_defaults_to_formal_configuration(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle, _ = self.make_bundle(Path(temp))
            with patch.object(runner.subprocess, "call", return_value=0) as call:
                self.assertEqual(runner.main(["--bundle", str(bundle), "--dry-run"]), 0)
            command = call.call_args.args[0]
            self.assertEqual(Path(command[command.index("--config") + 1]).name,
                             "swe_enclosure_h200_ep8_p256gb_8x.yaml")
            self.assertIn("--dry-run", command)
            self.assertEqual(command[command.index("--max-requests") + 1], "0")

    def test_reject_input_only_bundle_for_scene2_entry(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle, _ = self.make_bundle(Path(temp))
            manifest_path = bundle / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest.pop("hash_ids_scope")
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "input_output"):
                runner.validate_compact_bundle(bundle)


if __name__ == "__main__":
    unittest.main()
