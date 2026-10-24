import numpy as np
import pandas as pd
import torch
from torch import nn
from model import TransformerClassifier, add_physics_features
from runtime import tensor, require, fit, finish

# Stable, semantic labels replace nondeterministic os.listdir indexing.
FILES = [
    "healthyfirst2w.xlsx",
    "6.66-unhealthyfirst1w.xlsx",
    "13.33-unhealthysec1w.xlsx",
    "20-unhealthythir1w.xlsx",
]
COLUMNS = [
    "origin wind speed",
    "final wind",
    "peak current",
    "rotational speed",
    "Tambient",
    "TIGBT junction",
    "Tcase",
]


def run(a):
    series = []
    if a.demo:
        rng = np.random.default_rng(a.seed)
        for level in range(4):
            t = np.arange(1200) / 20
            current = 10 + 2 * np.sin(t)
            case = 30 + np.cos(t)
            series.append(
                np.column_stack(
                    [
                        10 + np.sin(t),
                        10 + np.sin(t),
                        current,
                        np.ones_like(t) * 100,
                        np.ones_like(t) * 20,
                        case + (1 + 0.2 * level) * current * 0.1,
                        case,
                    ]
                )
                + rng.normal(0, 0.02, (1200, 7))
            )
    else:
        for name in FILES:
            df = pd.read_excel(require(a.data_root, "data/" + name))
            series.append(df[COLUMNS].to_numpy(dtype=float))
    raw = [add_physics_features(x) for x in series]
    from sklearn.preprocessing import MinMaxScaler

    sx = MinMaxScaler().fit(np.concatenate([x[: int(len(x) * 0.7)] for x in raw]))

    def windows(start, stop):
        xs = []
        ys = []
        for label, x in enumerate(raw):
            x = sx.transform(x[int(len(x) * start) : int(len(x) * stop)])
            if len(x) < 100:
                raise ValueError("Each split needs at least 100 consecutive samples")
            ends = np.linspace(
                100,
                len(x),
                max(1, min(a.max_samples // 4, (len(x) - 100) // 10 + 1)),
                dtype=int,
            )
            for end in ends:
                xs.append(x[end - 100 : end])
                ys.append(label)
        return tensor(np.stack(xs)), torch.tensor(ys, dtype=torch.long)

    x, y = windows(0, 0.7)
    xv, yv = windows(0.8, 1)
    model = TransformerClassifier(10, 100, 32, 4, 2, 64, 0.1, 4)

    def loss(idx):
        ce = nn.functional.cross_entropy(model(x[idx]), y[idx])
        return ce, {"cross_entropy": ce}

    history = fit(model, loss, len(x), a)
    model.eval()
    with torch.no_grad():
        logits = model(xv)
        pred = logits.argmax(-1).numpy()
    return finish(
        model,
        a,
        history,
        pred,
        yv.numpy(),
        {
            "task": "degradation classification",
            "heldout_accuracy": float(np.mean(pred == yv.numpy())),
            "labels": FILES,
            "window": 100,
            "model_size": "compact execution configuration",
        },
        {"features": sx},
    )
