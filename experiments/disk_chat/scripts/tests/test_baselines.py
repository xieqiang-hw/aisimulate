"""Baseline identity, capacity, completion and safe resume checks."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import yaml
import run_baselines as baseline


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.template = yaml.safe_load(baseline.TEMPLATE.read_text())
        self.args = argparse.Namespace(speedup=1, memory_limit_gb=None, reserve_memory_gb=None)

    def test_capacity_units_and_reference_settings(self):
        result = baseline.configure(self.template, Path("/test/bundle"), 384, 8, self.args)
        kv = result["engine"]["workers"]["prefill"]["kv_cache"]
        self.assertEqual(kv["host_offload"]["num_host_blocks"], 12455)
        self.assertEqual(kv["g3_offload"]["num_g3_blocks"], 1816378)
        self.assertEqual(kv["g3_offload"]["read_bandwidth_gbps"], 8)
        self.assertEqual(result["traffic"]["source"]["paths"], ["/test/bundle/requests.jsonl"])
        self.assertEqual(result["engine"]["workers"]["prefill"]["scheduler"]["max_batched_tokens"], 16384)
        self.assertEqual(result["engine"]["workers"]["decode"]["scheduler"]["max_sequences"], 256)
        self.assertEqual(self.template["traffic"]["source"]["paths"], ["__SET_BY_RUNNER__"])

    def test_capacity_formula_rejects_different_parallelism(self):
        self.template["engine"]["workers"]["prefill"]["parallelism"]["tensor"] = 4
        with self.assertRaisesRegex(ValueError, "layout"):
            baseline.configure(self.template, Path("/test/bundle"), 384, 8, self.args)

    def test_speed_and_resources_are_preserved(self):
        self.args.speedup = 0.5
        self.args.memory_limit_gb = 20
        self.args.reserve_memory_gb = 4
        result = baseline.configure(self.template, Path("/test/bundle"), 64, 2, self.args)
        self.assertEqual(result["traffic"]["load"]["speedup"], 0.5)
        self.assertEqual(result["execution"]["resources"]["memory_limit_gb"], 20)
        self.assertEqual(result["execution"]["resources"]["reserve_memory_gb"], 4)

    def test_fingerprint_changes_with_execution_identity(self):
        first = baseline.canonical_hash({"config": self.template, "max_requests": 0})
        self.assertNotEqual(first, baseline.canonical_hash({"config": self.template, "max_requests": 5}))
        self.template["traffic"]["load"]["speedup"] = 2
        self.assertNotEqual(first, baseline.canonical_hash({"config": self.template, "max_requests": 0}))


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run_dir = self.root / "child"
        self.run_dir.mkdir()
        self.run_path = self.run_dir / "run.json"
        self.run = {"status": "completed", "returncode": 0, "dry_run": False, "selected_requests": 5}
        self.run_path.write_text(json.dumps(self.run))
        prediction = self.run_dir / "prediction"
        prediction.mkdir()
        self.prediction = prediction / "prediction.json"
        self.prediction.write_text(json.dumps({"completed_requests": 5, "per_request": []}))

    def tearDown(self):
        self.temp.cleanup()

    def test_real_completion_requires_all_selected_requests(self):
        case = {}
        baseline.record_child(case, self.root, 0, False)
        self.assertEqual(case["status"], "completed")
        self.prediction.write_text(json.dumps({"completed_requests": 4}))
        with self.assertRaisesRegex(ValueError, "only 4/5"):
            baseline.record_child({}, self.root, 0, False)

    def test_dry_run_is_not_a_completed_simulation(self):
        self.run.update(dry_run=True, status="prepared")
        self.run_path.write_text(json.dumps(self.run))
        case = {}
        baseline.record_child(case, self.root, 0, True)
        self.assertEqual(case["status"], "validated")
        with self.assertRaisesRegex(ValueError, "real replay"):
            baseline.record_child({}, self.root, 0, False)

    def test_resume_rejects_dry_run_marked_completed(self):
        self.run["dry_run"] = True
        self.run_path.write_text(json.dumps(self.run))
        sweep = self.root / "sweep.json"
        sweep.write_text(json.dumps({"schema": "chat-baseline-v1", "cases": [{
            "status": "completed", "run_json": str(self.run_path), "fingerprint": "case-a"}]}))
        with self.assertRaisesRegex(ValueError, "successful real replay"):
            baseline.reusable_cases([sweep])

    def test_resume_does_not_skip_failed_cases(self):
        sweep = self.root / "sweep.json"
        sweep.write_text(json.dumps({"schema": "chat-baseline-v1", "cases": [{
            "status": "failed", "fingerprint": "case-a"}]}))
        self.assertEqual(baseline.reusable_cases([sweep]), {})


if __name__ == "__main__":
    unittest.main()
