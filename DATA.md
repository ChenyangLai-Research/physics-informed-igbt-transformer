# Local data contract

No research data are bundled. Obtain access from the dataset owner; a publication DOI does not grant redistribution rights. `--data-root` is the root of the corresponding original project, not this repository.

Under `data/`: `healthyfirst2w.xlsx`, `6.66-unhealthyfirst1w.xlsx`, `13.33-unhealthysec1w.xlsx`, `20-unhealthythir1w.xlsx`. Required columns: origin wind speed, final wind, peak current, rotational speed, Tambient, TIGBT junction, Tcase. These exact names define a stable healthy/light/medium/high label order.

## Preprocessing and split

Each series is split chronologically: first 70% training, next 10% unused reserved validation, last 20% held out. Fit scaling on training blocks only. Length-100 windows; compact d_model=32, 4 heads, 2 layers, feedforward=64, dropout=0.1. Report accuracy and predicted class labels.

Missing files produce an actionable error. Dataset schemas were inspected locally; the synthetic demo only checks the code path and is not a substitute for research data.
