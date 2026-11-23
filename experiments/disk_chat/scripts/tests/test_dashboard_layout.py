"""Keep chat reports on the SWE template while preserving dataset isolation."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_chat_dashboard as dashboard


class SharedLayoutTests(unittest.TestCase):
    def test_shared_swe_layout_and_chart_sections_are_retained(self):
        page = dashboard.template()
        for name in ('capacityTab', 'readTab', 'compareTab', 'reuseChart', 'metricChart',
                     'stackChart', 'ssdReadChart', 'tableRows', 'scopeRows', 'datasetSelect'):
            self.assertIn(f'id="{name}"', page)
        self.assertEqual(page.count('__REPORT_DATA__'), 1)
        self.assertIn('function loadRows(){return datasetRows().filter', page)
        self.assertNotIn('旧分散报告存档在', page)

    def test_payload_keeps_dataset_identity_and_unrounded_values(self):
        source = dict(dataset='qwen-chat', status='已完成', status_code='completed',
                      p_dram_gb=384, read_gbps=8, budget=16384, speedup=1,
                      expected=2, sessions=1, span_seconds=10, arrival_kind='observed',
                      source_block_size=16, values={'sla':0.0052083,'completed':2},
                      run_json='/run/run.json', sweep='/run/sweep.json')
        result = dashboard.payload(dict(rows=[source], generated='2026-11-23', warnings=[],metrics=[('sla','SLA','%')]))
        self.assertEqual(result['rows'][0]['dataset'],'qwen-chat')
        self.assertEqual(result['rows'][0]['values']['sla'],0.0052083)
        self.assertEqual(result['rows'][0]['cells'][2],'0.005%')
        self.assertEqual(result['rows'][0]['cells'][3],'—')
        self.assertEqual(result['datasets']['qwen-chat']['requests'],2)

    def test_shared_template_anchor_drift_is_an_error(self):
        with self.assertRaisesRegex(ValueError,'anchor changed'):
            dashboard.replace_once('missing','anchor','new')


if __name__=='__main__':
    unittest.main()
