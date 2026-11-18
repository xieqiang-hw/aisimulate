import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from characterize_datasets import analyze, stats, bucket


class AnalysisTests(unittest.TestCase):
    def test_rank_and_tail(self):
        result = stats({0: 1, 10: 98, 100: 1})
        self.assertEqual(result['p50'], 10)
        self.assertEqual(result['p99'], 10)
        self.assertEqual(result['top1_share'], 100/1080)
        self.assertEqual(bucket(128), 1)

    def test_stream_pair_and_missing_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            rows=[dict(session_id='s', timestamp=t, input_length=x, output_length=2,
                       provider_tokens=dict(prompt=x,cached=None)) for t,x in [(0,10),(600000,20)]]
            raw=''.join(json.dumps(r)+'\n' for r in rows).encode()
            (root/'source_requests.jsonl').write_bytes(raw)
            (root/'requests.jsonl').write_bytes(raw)
            manifest=dict(request_count=2, files={'source_requests.jsonl':{'sha256':hashlib.sha256(raw).hexdigest()}})
            (root/'manifest.json').write_text(json.dumps(manifest))
            d=analyze(root,'FreeInference',300000)
            self.assertEqual(d['summary']['input']['total'],30)
            self.assertEqual(d['summary']['requests_per_window']['n'],3)
            self.assertEqual(d['cache_observed'],0)
            self.assertEqual(d['summary']['session_gap']['p50'],600000)
            self.assertEqual(analyze(root,'SWE-chat',300000)['n'],2)
            (root/'requests.jsonl').write_bytes(raw.splitlines(keepends=True)[0])
            with self.assertRaisesRegex(ValueError,'count mismatch'):
                analyze(root,'SWE-chat',300000)
            manifest['files']['source_requests.jsonl']['sha256']='invalid'
            (root/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'Checksum mismatch'):
                analyze(root,'FreeInference',300000)


if __name__=='__main__':
    unittest.main()
