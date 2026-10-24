# Validation — 2026-10-24

Local environment: CPython 3.12.14, macOS arm64, CPU. Each repository was installed into a fresh virtual environment from `requirements-tested.txt`. Pip was then upgraded to 26.2.2. `requirements-lock-macos-py312.txt` records the resolved application dependencies.

| Check | Local result |
|---|---|
| Fresh dependency installation and `python -m pip check` | Passed |
| Python syntax, including archival Python | Passed |
| Maintained-code Ruff E9/F63/F7/F82 checks | Passed |
| `python -m unittest discover -s tests -v` | 6 tests passed |
| Synthetic training, inference and checkpoint replay | Passed |
| Original local-data training, inference and checkpoint replay | Passed |
| Dependency advisory audit after pip upgrade | No known advisories reported by pip-audit at audit time |

Smoke commands use seed 22, `--max-samples 32` and 3 epochs. Replay predictions agree within rtol=1e-6 and atol=1e-7. Original data were read only from the owner-provided project directory; they are not bundled or uploaded.

```sh
python -m pip check
python -m compileall -q run.py runtime.py model.py experiment.py tests legacy
python -m unittest discover -s tests -v
python run.py --demo --epochs 3 --max-samples 32 --output outputs/audit-demo
python run.py --demo --epochs 3 --max-samples 32 --checkpoint outputs/audit-demo/model.pt --output outputs/audit-replay
python run.py --data-root "/path/to/authorized/project" --epochs 3 --max-samples 32 --output outputs/audit-real
```

Use a new output directory each time. The original-data path is described in DATA.md. Input filenames and private research metrics are retained in local audit records, not this document.

These checks validate execution, finite outputs and protocol consistency. They do not reproduce published accuracy, complete training, cross-validation, calibration or comparative studies.

The GitHub Actions matrix is defined for Python 3.12 on Linux, macOS and Windows. Defining or linting a workflow is not evidence of a successful hosted run. Hosted CI status is reported separately. CUDA/MPS, other Python versions, legacy experiment execution and MATLAB are unverified.

The earlier 2026-10-23 validation applies to the preceding revision. This audit adds input fingerprints, output protection and additional regressions; protocol version 1 checkpoints must be recreated.
