# Validation — 2026-10-24

Local environment: CPython 3.12.14, macOS arm64, CPU. Each repository was installed into a fresh virtual environment from `requirements-tested.txt`. Pip was then upgraded to 26.2.2. `requirements-lock-macos-py312.txt` records the resolved application dependencies.

| Check | Local result |
|---|---|
| Fresh dependency installation and `python -m pip check` | Passed |
| Python syntax, including archival Python | Passed |
| Maintained-code Ruff E9/F63/F7/F82 checks | Passed |
| `python -m unittest discover -s tests -v` | 8 tests passed |
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

The earlier 2026-10-23 validation applies to the preceding revision. This audit adds input fingerprints, output protection and additional regressions; protocol versions 1 and 2 checkpoints must be recreated.

## Follow-up verification — 2026-10-24

The updated source was rechecked locally: syntax, maintained-code lint, full regression suite, workflow lint, and original-data short training plus checkpoint replay all passed. The regression suite also covers synthetic training/replay, rejection of a different repository/source revision, legacy protocol rejection, and protocol portability without absolute data-root disclosure. See the test count above.

This round reused the independently installed per-repository environments from the initial audit; it did not claim ten additional fresh installations. Original-data runs used 3 epochs (12 for slurry), max-samples 32, and replay comparison rtol=1e-6/atol=1e-7. No original data were uploaded. Modified legacy code remains outside the full-experiment execution claim.

The preceding audit passed 30 hosted OS jobs. This PR now adds pip-audit 2.10.1 to each OS job; inspect checks at the current PR head for the new advisory results. Advisory checks depend on external service availability and the database state at run time. They are not proof that no vulnerability exists.

A macOS job in the ten-repository follow-up audit correctly failed on an inherited certifi 2026.7.22 installation (PYSEC-2026-4320 / GHSA-6q48-6c7w-9gw2). `requirements-dev.txt` now pins the audit toolchain and certifi 2026.10.19. This is a CI/tooling dependency correction, not a scientific model change. Check the final PR-head run after this correction.
