"""Exercise the bounded runner queue with real, lightweight child processes."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_baselines as baseline

FAKE_RUNNER = '''
import argparse,json,os,time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--config');p.add_argument('--output-root')
a,_=p.parse_known_args()
name=Path(a.config).stem
def event(kind):
 data=(json.dumps(dict(event=kind,case=name,time=time.monotonic(),pid=os.getpid()))+'\\n').encode()
 fd=os.open(os.environ['CHAT_TEST_EVENTS'],os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
 os.write(fd,data);os.close(fd)
failed='fail' in name
event('start');time.sleep(0.05 if failed else 0.35)
root=Path(a.output_root)/'child';(root/'prediction').mkdir(parents=True)
(root/'run.json').write_text(json.dumps(dict(status='failed' if failed else 'completed',returncode=7 if failed else 0,dry_run=False,selected_requests=1,wall_seconds=0.35)))
(root/'prediction/prediction.json').write_text(json.dumps(dict(completed_requests=1)))
event('end')
raise SystemExit(7 if failed else 0)
'''


class ParallelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.runner = self.root / 'fake_runner.py'
        self.runner.write_text(FAKE_RUNNER)
        self.events = self.root / 'events.jsonl'
        self.args = argparse.Namespace(
            skip_completed_from=[], output_root=self.root / 'results',
            template=baseline.TEMPLATE, dry_run=False, jobs=4, speedup=1,
            max_requests=1, capture_scheduler_trace=False, export_html=False,
            fail_fast=False,
        )

    def tearDown(self):
        self.temp.cleanup()

    def run_queue(self, fail=False):
        cases = [dict(name=f'case-{index}' + ('-fail' if fail and index == 0 else ''),
                      fingerprint=f'fingerprint-{index}', bundle=str(self.root),
                      config={'test_case': index}, status='pending') for index in range(8)]
        with patch.object(baseline, 'RUNNER', self.runner), patch.dict(os.environ, {'CHAT_TEST_EVENTS': str(self.events)}):
            code = baseline.execute(cases, self.args)
        plan = json.loads(next((self.root / 'results').glob('*/sweep.json')).read_text())
        return code, plan

    def test_four_real_children_overlap_and_queue_never_exceeds_four(self):
        code, plan = self.run_queue()
        self.assertEqual(code, 0)
        self.assertEqual(plan['jobs'], 4)
        self.assertEqual(plan['max_active_runners'], 4)
        self.assertTrue(all(case['status'] == 'completed' for case in plan['cases']))
        events = sorted([json.loads(line) for line in self.events.read_text().splitlines()], key=lambda value: value['time'])
        active = maximum = 0
        for event in events:
            active += 1 if event['event'] == 'start' else -1
            maximum = max(active, maximum)
            self.assertGreaterEqual(active, 0)
            self.assertLessEqual(active, 4)
        self.assertEqual(maximum, 4)
        self.assertEqual(active, 0)

    def test_failed_child_is_recorded_and_other_cases_continue(self):
        code, plan = self.run_queue(fail=True)
        self.assertEqual(code, 1)
        self.assertEqual(sum(case['status'] == 'failed' for case in plan['cases']), 1)
        self.assertEqual(sum(case['status'] == 'completed' for case in plan['cases']), 7)

    def test_fail_fast_stops_new_launches_but_reaps_active_children(self):
        self.args.fail_fast = True
        code, plan = self.run_queue(fail=True)
        self.assertEqual(code, 1)
        self.assertEqual(sum(case['status'] == 'pending' for case in plan['cases']), 4)
        self.assertEqual(sum(case['status'] == 'running' for case in plan['cases']), 0)
        self.assertIn('halt_reason', plan)

    def test_interrupt_reaps_all_four_children_and_saves_status(self):
        real_sleep = baseline.time.sleep
        calls = 0

        def interrupt_once(seconds):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise KeyboardInterrupt
            return real_sleep(seconds)

        with patch.object(baseline.time, 'sleep', side_effect=interrupt_once):
            code, plan = self.run_queue()
        self.assertEqual(code, 130)
        interrupted = [case for case in plan['cases'] if case['status'] == 'interrupted']
        self.assertEqual(len(interrupted), 4)
        self.assertFalse(any(case['status'] == 'running' for case in plan['cases']))
        for case in interrupted:
            with self.assertRaises(ProcessLookupError):
                os.kill(case['pid'], 0)


if __name__ == '__main__':
    unittest.main()
