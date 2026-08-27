#!/usr/bin/env python3
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/qwen_builder_preflight.py"


class BuilderPreflightTests(unittest.TestCase):
    def test_report_is_read_only_and_machine_readable(self):
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "record.json"
            run = subprocess.run(
                ["python3", str(SCRIPT), "--output", str(output)],
                check=True, capture_output=True, text=True,
            )
            stdout = json.loads(run.stdout)
            written = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(stdout, written)
            self.assertEqual(written["schema"],
                             "ember.qwen3.8.builder-preflight.v1")
            self.assertFalse(written["mutated"])
            self.assertEqual(len(written["source_revision"]), 40)
            self.assertGreater(written["memory"]["ram_bytes"], 0)
            self.assertIsInstance(written["filesystems"], list)
            self.assertNotIn("token", run.stdout.lower())


if __name__ == "__main__":
    unittest.main()
