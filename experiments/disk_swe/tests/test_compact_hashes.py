"""Behavioral checks for the exact-to-compact Mooncake conversion."""
import importlib.util
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/compact_swe_chat_hashes.py"
spec = importlib.util.spec_from_file_location("compact_swe_chat_hashes", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CompactHashesTest(unittest.TestCase):
    def test_split_output_and_later_input_have_identical_complete_prefix(self):
        prompt = list(range(100))
        output = list(range(100, 170))
        first = module.convert_row({
            "input_length": 100, "output_length": 70,
            "input_token_ids": prompt, "output_token_ids": output,
            "timestamp": 50, "session_id": "s1", "delay": 12, "priority": 3,
        })
        second = module.convert_row({
            "input_length": 160, "output_length": 0,
            "input_token_ids": prompt + output[:60], "output_token_ids": [],
            "timestamp": 80,
        })
        self.assertEqual(first["hash_ids"][:10], second["hash_ids"])
        self.assertNotIn("input_token_ids", first)
        self.assertNotIn("output_token_ids", first)
        self.assertEqual(first["delay"], 12)
        self.assertEqual(first["session_id"], "s1")
        self.assertEqual(first["priority"], 3)

    def test_different_last_token_only_invalidates_its_prefix_and_descendants(self):
        original = list(range(192))
        changed = original.copy()
        changed[127] = 999
        a = module.combined_hash_ids(original)
        b = module.combined_hash_ids(changed)
        self.assertEqual(a[:7], b[:7])
        self.assertNotEqual(a[7], b[7])
        self.assertNotEqual(a[8], b[8])

    def test_reject_missing_or_invalid_exact_token_plan(self):
        row = {"input_length": 1, "output_length": 1,
               "input_token_ids": [3], "output_token_ids": [4]}
        for field, bad in (("input_token_ids", []),
                           ("output_token_ids", [-1]),
                           ("output_token_ids", [2 ** 32])):
            with self.assertRaises(ValueError):
                module.convert_row({**row, field: bad})


if __name__ == "__main__":
    unittest.main()
