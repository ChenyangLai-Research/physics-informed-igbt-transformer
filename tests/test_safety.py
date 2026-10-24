"""Regression tests for input validation, replay integrity and output protection."""

from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import runtime


class SafetyTests(unittest.TestCase):
    def test_reject_nonfinite_and_empty_inputs(self):
        good = np.array([[1.0, 2.0], [3.0, 4.0]])
        for bad in [
            np.empty((0, 2)),
            np.array([[np.nan, 2.0]]),
            np.array([[np.inf, 2.0]]),
        ]:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    runtime.normalize(good, bad)

    def test_replay_detects_changed_file_contents(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            file = root / "sample.txt"
            file.write_text("original sample")
            runtime._INPUTS.clear()
            runtime.require(root, file.name)
            args = SimpleNamespace(checkpoint=True, saved_inputs=dict(runtime._INPUTS))
            runtime.verify_replay(args)
            file.write_text("different sample")
            runtime.require(root, file.name)
            with self.assertRaisesRegex(ValueError, "fingerprints"):
                runtime.verify_replay(args)
            runtime._INPUTS.clear()

    def test_cli_rejects_invalid_values(self):
        with tempfile.TemporaryDirectory() as folder:
            for flags in [
                ["--lr", "nan"],
                ["--lr", "0"],
                ["--seed", "-1"],
                ["--max-samples", "15"],
            ]:
                with self.subTest(flags=flags):
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(ROOT / "run.py"),
                            "--demo",
                            "--output",
                            str(Path(folder) / "new"),
                            *flags,
                        ],
                        cwd=folder,
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    self.assertEqual(
                        result.returncode, 2, result.stdout + result.stderr
                    )
                    self.assertNotIn("Traceback", result.stderr)

    def test_existing_outputs_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            sentinel = Path(folder) / "keep.txt"
            sentinel.write_text("keep this result")
            result = subprocess.run(
                [sys.executable, str(ROOT / "run.py"), "--demo", "--output", folder],
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("will not be overwritten", result.stderr)
            self.assertEqual(sentinel.read_text(), "keep this result")


if __name__ == "__main__":
    unittest.main()
