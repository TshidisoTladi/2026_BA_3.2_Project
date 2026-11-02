"""
tune_models.py - Step 4: improve model accuracy with hyper-parameter tuning

Uses RandomizedSearchCV + TimeSeriesSplit (train on the past, validate on the future)
on the TRAINING days only. The tuned models are then scored ONCE on the test days
and compared with the untuned versions from train_regression.py / train_classification.py.
If a tuned model is better, it replaces the saved one.
"""
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

from features import (CLASSIFICATION_FEATURES, CLASSIFICATION_TARGET,
                      REGRESSION_FEATURES, REGRESSION_TARGET, SITUATION_LABELS,
                      build_dataset, chronological_split)

ROOT = Path(__file__).resolve().parent.parent
tscv = TimeSeriesSplit(n_splits=5)

data = build_dataset()
train, test = chronological_split(data)

# =============================================================== REGRESSION
print("=" * 60, "\nTUNING THE REGRESSION MODEL (Random Forest)\n" + "=" * 60)
X_tr, y_tr = train[REGRESSION_FEATURES], train[REGRESSION_TARGET]
X_te, y_te = test[REGRESSION_FEATURES], test[REGRESSION_TARGET]

old = joblib.load(ROOT / "models" / "regression_model.joblib")
old_pred = old["model"].predict(X_te)
old_mae = mean_absolute_error(y_te, old_pred)

search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_distributions={
        "n_estimators": randint(100, 400),
        "max_depth": [6, 8, 10, 14, None],
        "min_samples_leaf": randint(2, 25),
        "max_features": [0.5, 0.7, 1.0],
    },
    n_iter=20, cv=tscv, scoring="neg_mean_absolute_error",
    random_state=42, n_jobs=1,
)
search.fit(X_tr, y_tr)
print("Best parameters      :", search.best_params_)
print(f"Best CV MAE          : {-search.best_score_:.2f}")

new_pred = search.best_estimator_.predict(X_te)
new_mae = mean_absolute_error(y_te, new_pred)
print(f"Test MAE  before -> after tuning : {old_mae:.2f} -> {new_mae:.2f}")
print(f"Test R2   before -> after tuning : {r2_score(y_te, old_pred):.3f} -> "
      f"{r2_score(y_te, new_pred):.3f}")

if new_mae < old_mae:
    joblib.dump({"model": search.best_estimator_, "features": REGRESSION_FEATURES,
                 "name": "Random Forest (tuned)"},
                ROOT / "models" / "regression_model.joblib", compress=3)
    print("-> Tuned model is better and has been saved.")
else:
    print("-> Tuning did not help on the test days; the original model is kept.")

# ============================================================ CLASSIFICATION
print("\n" + "=" * 60, "\nTUNING THE CLASSIFICATION MODEL (Decision Tree)\n" + "=" * 60)
Xc_tr, yc_tr = train[CLASSIFICATION_FEATURES], train[CLASSIFICATION_TARGET].astype(int)
Xc_te, yc_te = test[CLASSIFICATION_FEATURES], test[CLASSIFICATION_TARGET].astype(int)

old_c = joblib.load(ROOT / "models" / "classification_model.joblib")
old_cpred = old_c["model"].predict(Xc_te)
old_f1 = f1_score(yc_te, old_cpred, average="macro")

csearch = RandomizedSearchCV(
    DecisionTreeClassifier(class_weight="balanced", random_state=42),
    param_distributions={
        "max_depth": randint(3, 16),
        "min_samples_leaf": randint(2, 30),
        "criterion": ["gini", "entropy"],
    },
    n_iter=30, cv=tscv, scoring="f1_macro", random_state=42, n_jobs=1,
)
csearch.fit(Xc_tr, yc_tr)
print("Best parameters      :", csearch.best_params_)
print(f"Best CV macro-F1     : {csearch.best_score_:.3f}")

new_cpred = csearch.best_estimator_.predict(Xc_te)
new_f1 = f1_score(yc_te, new_cpred, average="macro")
print(f"Test macro-F1 before -> after : {old_f1:.3f} -> {new_f1:.3f}")
print(f"Test accuracy before -> after : {accuracy_score(yc_te, old_cpred):.3f} -> "
      f"{accuracy_score(yc_te, new_cpred):.3f}")

if new_f1 > old_f1:
    joblib.dump({"model": csearch.best_estimator_, "features": CLASSIFICATION_FEATURES,
                 "labels": SITUATION_LABELS, "name": "Decision Tree (tuned)"},
                ROOT / "models" / "classification_model.joblib")
    print("-> Tuned model is better and has been saved.")
else:
    print("-> Tuning did not help on the test days; the original model is kept.")
