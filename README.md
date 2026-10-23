# Physics-informed Transformer for IGBT degradation assessment

Private working repository for Chenyang Lai's research code. [Associated publication](https://doi.org/10.3850/ESREL2026061419_esrel26-p30436-cd)

This is a **runnable, curated recovery** of selected legacy code. The default experiment is deliberately small and checks data loading, training, held-out prediction and checkpoint saving. It does **not** reproduce the paper's numerical results.

## Method

Seven measured signals are augmented with deltaT = Tjunction - Tcase, deltaT/(I+epsilon), and deltaT/(I²+epsilon). Transformer encoders with learned positions and mean/max temporal pooling classify four degradation levels.

## Install and run

Tested with Python 3.12 on macOS, CPU. From the repository directory:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-tested.txt
python run.py --demo --epochs 3 --max-samples 32 --output outputs/demo
```

The synthetic demonstration requires no research dataset. For authorized local research data, point to the original project root described in [DATA.md](DATA.md):

```sh
python run.py --data-root "/absolute/path/to/original/project" --epochs 3 --max-samples 32 --output outputs/real
```

`--epochs`, `--batch-size`, `--lr`, `--seed`, `--max-samples` and `--output` are configurable. Some models have additional fixed caps documented below. Longer training alone does not restore the full published protocol.

The output folder contains metrics/history, predictions and a checkpoint (`model.pt`); preprocessing information is saved separately or inside the joblib bundle. Outputs and all data files are git-ignored. Regression MSE is in normalized coordinates unless the task says otherwise.

To replay held-out inference from a checkpoint (not resume training):

```sh
python run.py --demo --seed 22 --max-samples 32 --checkpoint outputs/demo/model.pt --output outputs/replay
```

Replay requires the sibling `protocol.json`, identical data, data root/mode, seed and sample cap. The pipeline rebuilds its deterministic preprocessing and data split; this is not an arbitrary new-data prediction API. Only load trusted checkpoints.

## Data and execution protocol

Each series is split chronologically: first 70% training, next 10% unused reserved validation, last 20% held out. Fit scaling on training blocks only. Length-100 windows; compact d_model=32, 4 heads, 2 layers, feedforward=64, dropout=0.1. Report accuracy and predicted class labels.

## Validation

```sh
python -m unittest discover -s tests -v
```

The regression test trains on synthetic data, reloads the checkpoint and checks identical predictions. It also checks that mismatched replay settings and missing local data fail clearly. [VALIDATION.md](VALIDATION.md) records local-data checks without including the underlying data.

## Scope and limitations

The paper uses a larger model, many more windows, scheduling, early stopping and baseline/SHAP studies. Those experiments have not been reproduced by this compact entry point. The common heldout_mse field compares class indices and is not a classification-quality metric; use heldout_accuracy.

## Repository layout and provenance

- `run.py`: command-line entry point.
- `experiment.py`: data selection, training objective and evaluation protocol.
- `model.py`: extracted model definitions or explicitly documented port.
- `runtime.py`: shared CPU experiment utilities.
- `legacy/`: selected archival source for traceability; notebooks have outputs and execution metadata removed. These old scripts are reference material and may retain machine-specific paths and obsolete dependencies; run `run.py` instead.
- `project.json`: original relative source paths and original SHA-256 hashes. Notebook hashes refer to the original unsanitized source.
- `CHANGELOG.md`: changes and deviations from the archive.

Original folders were left unchanged. No raw data, manuscript, trained research weights or partner documents are distributed. No open-source license has been assigned; see [RIGHTS.md](RIGHTS.md). Keep this repository private until its scope and rights are reviewed.
