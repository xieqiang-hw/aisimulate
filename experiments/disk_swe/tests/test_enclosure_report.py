"""Report correctness: token weighting, waiting metrics and incomplete runs."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import yaml

BASE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("enclosure_report", BASE / "scripts/build_enclosure_report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class EnclosureReportTest(unittest.TestCase):
    def make_prediction(self, root):
        prediction = root / "prediction"
        prediction.mkdir()
        summary = {"completed_requests": 2, "goodput_completed_requests": 1,
                   "mean_ttft_ms": 1000, "p99_ttft_ms": 2000, "mean_tpot_ms": 10,
                   "g3_offload": {"read": {"completed_bytes": 2e12},
                                 "write": {"completed_bytes": 3e12},
                                 "decode_writeback_blocks": 7, "prefill_decode_reuse_blocks": 4},
                   "per_request": []}
        (prediction / "prediction.json").write_text(json.dumps(summary, sort_keys=True))
        rows = []
        for length, g1, g2, g3, wait in [(100, 40, 20, 10, 100), (300, 60, 100, 80, 900)]:
            rows.append({"input_length": length,
                         report.TOKEN_KEYS[0]: g1, report.TOKEN_KEYS[1]: g2,
                         report.TOKEN_KEYS[2]: g3,
                         "prefill_wait": {"ttft_excluding_prefill_scheduling_ms": wait}})
        (prediction / "requests.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        (prediction / "resource-runtime.json").write_text(json.dumps({"peak_observed_rss_bytes": 1e9}))
        return root

    def test_completed_report_preserves_tier_and_io_units(self):
        with tempfile.TemporaryDirectory() as temp:
            values = report.completed_metrics(self.make_prediction(Path(temp)), 2)
        self.assertEqual(values["hbm"], 25)
        self.assertEqual(values["dram"], 30)
        self.assertEqual(values["enclosure"], 22.5)
        self.assertEqual(values["reuse"], 77.5)
        self.assertEqual(values["sla"], 50)
        self.assertEqual(values["adjusted_mean"], .5)
        self.assertEqual(values["adjusted_p99"], .9)
        self.assertEqual(values["enclosure_read"], 2)
        self.assertEqual(values["enclosure_write"], 3)
        self.assertEqual(values["pd_reuse"], 4)
        self.assertEqual(values["memory"], 1)
        row = {"values": values, "status": "已完成", "wall_seconds": 10}
        self.assertEqual(len(report.cells(row)), len(report.COLUMNS))

    def test_incomplete_full_run_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.make_prediction(Path(temp))
            with self.assertRaisesRegex(ValueError, "count mismatch"):
                report.completed_metrics(root, 3)

    def test_successful_retry_supersedes_failure_but_keeps_history(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = root / "bundle"
            bundle.mkdir()
            (bundle / "manifest.json").write_text(json.dumps({
                "hash_ids_scope": "input_output", "request_count": 2}))
            config = (BASE / "configs/swe_enclosure_h200_ep8_p256gb_8x.yaml").read_text()
            for name, status in [("001", "failed"), ("002", "completed")]:
                run = root / name
                run.mkdir()
                (run / "run.json").write_text(json.dumps({
                    "bundle": str(bundle), "status": status,
                    "max_requests": 0, "selected_requests": 2}))
                (run / "resolved.yaml").write_text(config)
                if status == "completed":
                    self.make_prediction(run)
            rows = report.collect(root, {})
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["id"], "002")
            self.assertEqual(rows[0]["status"], "已完成")
            self.assertEqual([a["status"] for a in rows[0]["attempts"]], ["失败", "已完成"])

    def test_running_row_does_not_read_partial_prediction(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = root / "bundle"
            bundle.mkdir()
            (bundle / "manifest.json").write_text(json.dumps({
                "hash_ids_scope": "input_output", "request_count": 2}))
            run = root / "run"
            run.mkdir()
            (run / "run.json").write_text(json.dumps({
                "bundle": str(bundle), "status": "running", "max_requests": 0}))
            config = yaml.safe_load((BASE / "configs/swe_enclosure_h200_ep8_p256gb_8x.yaml").read_text())
            (run / "resolved.yaml").write_text(yaml.safe_dump(config))
            (run / "prediction").mkdir()
            (run / "prediction/prediction.json").write_text("{incomplete")
            rows = report.collect(root, {})
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "运行中")
            self.assertEqual(rows[0]["values"], {})
            self.assertTrue(all(cell == "—" for cell in rows[0]["cells"][1:]))


if __name__ == "__main__":
    unittest.main()
