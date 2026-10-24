# Recovery notes — 2026-10-23

Replaced os.listdir-dependent labels with a semantic filename list and used disjoint temporal blocks. Kept original physics feature extraction and Transformer model family.

## Shared changes

Added an explicit data root, deterministic CPU seeds, dependency declarations, synthetic demonstration, finite loss/gradient/prediction checks, held-out evaluation, checkpoint replay with protocol checks, and source provenance. Train/test preprocessing is separated. Archival code is not executed at import.

## Deviations and unverified parts

The paper uses a larger model, many more windows, scheduling, early stopping and baseline/SHAP studies. Those experiments have not been reproduced by this compact entry point. The common heldout_mse field compares class indices and is not a classification-quality metric; use heldout_accuracy.

## 2026-10-24 release-readiness audit

- Specify Python 3.12, update the pip bootstrap, align default dependency installation with tested pins, and record a macOS transitive snapshot.
- Validate CLI values, reject empty/nonfinite normalization inputs, protect existing outputs, and record environment metadata.
- Add version-2 checkpoint protocols with input-file fingerprints. Older checkpoint protocols require retraining.
- Add regression tests and a CPU CI matrix; CI execution status is reported separately from local tests.
- Clarify runnable versus archival code, data confidentiality, reproducibility limits and release rights in the documentation.
