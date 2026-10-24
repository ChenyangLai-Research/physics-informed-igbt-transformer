"""Replay privacy and provenance regressions."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import runtime


class ReleaseSafetyTests(unittest.TestCase):
    def test_protocol_is_portable_and_excludes_machine_paths(self):
        args = SimpleNamespace(
            demo=False, seed=22, max_samples=32, data_root=Path("/private/alice/data")
        )
        first = runtime.build_protocol(args)
        args.data_root = Path("/different/machine/data")
        self.assertEqual(first, runtime.build_protocol(args))
        self.assertNotIn("data_root", first)
        self.assertNotIn("/private/", json.dumps(first))
        with tempfile.TemporaryDirectory() as folder:
            for location in ("original", "relocated"):
                root = Path(folder) / location
                root.mkdir()
                (root / "sample.txt").write_text("identical data")
                runtime._INPUTS.clear()
                runtime.require(root, "sample.txt")
                if location == "original":
                    inputs = dict(runtime._INPUTS)
                else:
                    runtime.verify_replay(
                        SimpleNamespace(checkpoint=True, saved_inputs=inputs)
                    )
        runtime._INPUTS.clear()

    def test_replay_rejects_wrong_repository_code_and_old_protocol(self):
        args = SimpleNamespace(demo=True, seed=22, max_samples=32)
        expected = runtime.build_protocol(args)
        saved = {**copy.deepcopy(expected), "inputs": {}}
        self.assertEqual(runtime.validate_protocol(saved, expected), {})
        for field, value in [
            ("repository", "different-project"),
            ("version", 2),
            ("code_sha256", {}),
        ]:
            bad = copy.deepcopy(saved)
            bad[field] = value
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "same repository"),
            ):
                runtime.validate_protocol(bad, expected)
        with self.assertRaisesRegex(ValueError, "fingerprints"):
            runtime.validate_protocol({**expected, "inputs": None}, expected)
