"""Execution, checkpoint replay and preprocessing regressions."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime import normalize


class RegressionTest(unittest.TestCase):
    def test_train_only_normalization(self):
        train = np.array([[1.0, 2.0], [3.0, 4.0]])
        heldout = np.array([[101.0, 202.0]])
        _, scaled, scaler = normalize(train, heldout)
        np.testing.assert_array_equal(scaler.data_max_, [3, 4])
        self.assertTrue((scaled > 1).all())

    def test_training_and_checkpoint_replay(self):
        # Keep temporary run artefacts under this repository, never the system temp dir.
        (ROOT / "outputs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as folder:
            folder = Path(folder)
            first = folder / "train"
            replay = folder / "replay"
            args = [
                sys.executable,
                "run.py",
                "--demo",
                "--epochs",
                "3",
                "--max-samples",
                "16",
            ]
            trained = subprocess.run(
                args + ["--output", str(first)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=180,
            )
            self.assertEqual(trained.returncode, 0, trained.stdout + trained.stderr)
            metrics = json.loads((first / "metrics.json").read_text())
            self.assertTrue(np.isfinite(metrics["heldout_mse"]))
            checkpoint = str(first / "model.pt")
            loaded = subprocess.run(
                args + ["--checkpoint", checkpoint, "--output", str(replay)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=180,
            )
            self.assertEqual(loaded.returncode, 0, loaded.stdout + loaded.stderr)
            np.testing.assert_allclose(
                np.load(first / "predictions.npz")["prediction"],
                np.load(replay / "predictions.npz")["prediction"],
                rtol=1e-6,
                atol=1e-7,
            )
            mismatch = subprocess.run(
                args
                + [
                    "--seed",
                    "99",
                    "--checkpoint",
                    checkpoint,
                    "--output",
                    str(folder / "bad"),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(mismatch.returncode, 2)
            self.assertIn("same repository", mismatch.stderr)
            missing = subprocess.run(
                [sys.executable, "run.py", "--data-root", str(folder / "nonexistent")],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(missing.returncode, 2, missing.stdout + missing.stderr)


if __name__ == "__main__":
    unittest.main()
