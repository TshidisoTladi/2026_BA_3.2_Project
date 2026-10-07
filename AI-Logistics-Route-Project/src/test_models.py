"""
test_models.py - quick check that the saved models load and make predictions.
Run this after training. It replays one row from the unseen test days.
"""
from pathlib import Path

import joblib

from features import (CLASSIFICATION_FEATURES, REGRESSION_FEATURES,
                      build_dataset, chronological_split)

ROOT = Path(__file__).resolve().parent.parent

reg = joblib.load(ROOT / "models" / "regression_model.joblib")
clf = joblib.load(ROOT / "models" / "classification_model.joblib")

_, test = chronological_split(build_dataset())
row = test.iloc[[100]]  # any row from the test days

predicted_volume = reg["model"].predict(row[REGRESSION_FEATURES])[0]
predicted_label = clf["labels"][int(clf["model"].predict(row[CLASSIFICATION_FEATURES])[0])]

print(f"Regression model ({reg['name']}):")
print(f"  predicted volume = {predicted_volume:.0f} vehicles | actual = {row['total'].iloc[0]}")
print(f"Classification model ({clf['name']}):")
print(f"  predicted label  = {predicted_label} | actual = {row['traffic_situation'].iloc[0]}")
