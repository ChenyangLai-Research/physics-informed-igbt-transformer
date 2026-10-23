# Validation — 2026-10-23

Environment: Python 3.12, macOS, CPU, versions in `requirements-tested.txt`.

- Synthetic demonstration: training, finite gradients/predictions and checkpoint saving.
- Original local dataset: 3 epochs, `--max-samples 32`, seed 22; data loaded directly from the user-supplied original project root. Completed successfully.
- This is an execution check. Published accuracy, full training, cross-validation and comparisons have **not** been reproduced.

The exact local command and logs are retained in the owner's `personal_web/qa/` directory. Research metrics and data are not shipped in this repository.

Regression suite: **2 tests passed**. The integration test covers synthetic training, checkpoint replay with equal predictions, rejection of different replay settings, and missing-data failure; the other test verifies training-only normalization.
