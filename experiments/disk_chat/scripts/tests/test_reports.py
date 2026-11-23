"""Check result arithmetic and prevent smoke/partial data from becoming results."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import refresh_experiments as report


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def requests(self, rows):
        path = self.root / "requests.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return path

    def row(self, length, g1, g2, g3, adjusted):
        return {"terminal_status": "completed", "input_length": length,
                "first_admission_g1_reused_input_tokens": g1,
                "first_admission_g2_direct_reused_input_tokens": g2,
                "first_admission_g3_reused_input_tokens": g3,
                "prefill_wait": {"ttft_excluding_prefill_scheduling_ms": adjusted}}

    def test_reuse_is_token_weighted_not_request_average(self):
        path = self.requests([self.row(100, 50, 20, 10, 100), self.row(900, 0, 0, 0, 300)])
        result = report.request_metrics(path, 2)
        self.assertEqual(result["hbm"], 5)
        self.assertEqual(result["reuse"], 8)
        self.assertAlmostEqual(result["adjusted_mean"], 0.2)
        self.assertAlmostEqual(result["adjusted_p99"], 0.3)

    def test_missing_metrics_stay_missing(self):
        row = self.row(100, 50, 20, 10, 100)
        del row["first_admission_g2_direct_reused_input_tokens"]
        del row["prefill_wait"]
        result = report.request_metrics(self.requests([row]), 1)
        self.assertIsNone(result["dram"])
        self.assertIsNone(result["reuse"])
        self.assertIsNone(result["adjusted_mean"])

    def test_partial_detail_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "detail count"):
            report.request_metrics(self.requests([self.row(100, 0, 0, 0, 100)]), 2)

    def test_summary_reader_ignores_large_detail_list(self):
        path = self.root / "prediction.json"
        path.write_text('{"completed_requests": 2, "mean_ttft_ms": 10, "per_request": [' + ' ' * 200000 + ']}')
        self.assertEqual(report.summary_prefix(path), {"completed_requests": 2, "mean_ttft_ms": 10})

    def test_swe_nearest_rank_percentile(self):
        self.assertEqual(report.percentile99(list(range(101))), 99)

    def test_small_nonzero_sla_is_not_formatted_as_zero(self):
        self.assertEqual(report.result_cell({"values": {"sla": 0.0052083}}, "sla"), "0.005")
        self.assertIsNone(report.result_cell({"values": {}}, "sla"))

    def test_smoke_and_dry_run_plans_are_excluded(self):
        paths = []
        for n, dry_run, limit in [(0, True, 0), (1, False, 3)]:
            path = self.root / f"{n}.json"
            path.write_text(json.dumps({"schema": "chat-baseline-v1", "dry_run": dry_run,
                                        "max_requests": limit, "cases": [{"status": "completed"}]}))
            paths.append(path)
        rows, warnings, sources = report.collect(paths, self.root)
        self.assertEqual(rows, [])
        self.assertEqual(warnings, [])
        self.assertEqual(sources, [])


if __name__ == "__main__":
    unittest.main()
