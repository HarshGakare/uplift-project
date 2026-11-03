"""
Check: do the app's features match the features used in training?
Run from the project root:  python src/check_features.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from features import build_features

PROC_DIR = Path("data/processed")

with open(PROC_DIR / "feature_cols.json") as f:
    feature_cols = json.load(f)

raw = pd.read_csv("data/hillstrom.csv")
app_features = build_features(raw, feature_cols)

processed = pd.concat(
    [pd.read_csv(PROC_DIR / "train.csv"), pd.read_csv(PROC_DIR / "test.csv")]
)[feature_cols]

# The split shuffled the rows, so we sort both tables the same way
a = app_features.sort_values(feature_cols).reset_index(drop=True)
b = processed.sort_values(feature_cols).reset_index(drop=True)

same_shape = a.shape == b.shape
same_values = same_shape and np.allclose(a.to_numpy(), b.to_numpy())
print("Same shape:", same_shape, a.shape, b.shape)
print("Same values:", same_values)
print("\nFEATURES MATCH TRAINING" if same_values else "\nMISMATCH: fix features.py before using the app")