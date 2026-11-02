"""
train_classification.py - Step 4: Train the CLASSIFICATION model

Goal   : label the CURRENT state of a road segment as low / normal / high / heavy.
Why    : the label is what gets colour-coded on the route map, and gives the driver
         an easy-to-read warning next to the regression model's number.
Method : live vehicle counts + context as inputs, chronological split,
         time-series cross-validation, class weights for the rare classes.
"""
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from features import (CLASSIFICATION_FEATURES, CLASSIFICATION_TARGET,
                      REGRESSION_FEATURES, SITUATION_LABELS, build_dataset,
                      chronological_split)

ROOT = Path(__file__).resolve().parent.parent
(ROOT / "models").mkdir(exist_ok=True)
(ROOT / "outputs").mkdir(exist_ok=True)

# ---------------------------------------------------------------- 1. data
data = build_dataset()
train, test = chronological_split(data, test_fraction=0.2)
X_train, y_train = train[CLASSIFICATION_FEATURES], train[CLASSIFICATION_TARGET].astype(int)
X_test, y_test = test[CLASSIFICATION_FEATURES], test[CLASSIFICATION_TARGET].astype(int)
print(f"Train: {len(train)} rows | Test: {len(test)} rows")
print("Class balance in training data:")
print(train["traffic_situation"].value_counts(normalize=True).round(3).to_string(), "\n")

# ------------------------------------------------- 2. candidate models
candidates = {
    "Baseline (most common class)": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced")),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=8, min_samples_leaf=10, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, min_samples_leaf=3, class_weight="balanced",
        n_jobs=-1, random_state=42),
}

# ------------------------------------------------- 3. cross-validation
tscv = TimeSeriesSplit(n_splits=5)
cv_scores = {}
print("Cross-validation on the training days (macro F1, higher is better):")
for name, model in candidates.items():
    start = time.time()
    scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring="f1_macro")
    cv_scores[name] = scores.mean()
    print(f"  {name:<30} CV macro-F1 = {scores.mean():.3f}  (+/- {scores.std():.3f})"
          f"   [{time.time() - start:.1f}s]")

# ------------------------------------------------- 4. pick + final fit
real_models = {k: v for k, v in cv_scores.items() if not k.startswith("Baseline")}
best_name = max(real_models, key=real_models.get)
print(f"\nBest model by cross-validation: {best_name}")

# ------------------------------------------------- 5. evaluate on TEST days
rows = []
for name, model in candidates.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rows.append({
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "macro_F1": f1_score(y_test, pred, average="macro"),
        "weighted_F1": f1_score(y_test, pred, average="weighted"),
    })
results = pd.DataFrame(rows).round(3)
print("\nTest-set results (unseen days):")
print(results.to_string(index=False))
results.to_csv(ROOT / "outputs" / "classification_results.csv", index=False)

best_model = candidates[best_name]
best_pred = best_model.predict(X_test)
print(f"\nDetailed report for {best_name}:")
print(classification_report(y_test, best_pred, labels=[0, 1, 2, 3],
                            target_names=SITUATION_LABELS, zero_division=0))

# ------------------------------------------------- 5b. honest check: forecast mode
# The label in this dataset is derived from the vehicle counts, so a high score with
# live counts is expected. What happens if we take the live counts AWAY and only use
# context + past traffic (i.e. predicting upcoming congestion)?
forecast_rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                     class_weight="balanced", n_jobs=-1, random_state=42)
forecast_rf.fit(train[REGRESSION_FEATURES], y_train)
fpred = forecast_rf.predict(test[REGRESSION_FEATURES])
print("Forecast-mode check (no live counts, Random Forest): "
      f"accuracy = {accuracy_score(y_test, fpred):.3f}, "
      f"macro-F1 = {f1_score(y_test, fpred, average='macro'):.3f}\n")

# ------------------------------------------------- 6. save model + plot
joblib.dump({"model": best_model, "features": CLASSIFICATION_FEATURES,
             "labels": SITUATION_LABELS, "name": best_name},
            ROOT / "models" / "classification_model.joblib")
print("Saved -> models/classification_model.joblib")

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, best_pred, labels=[0, 1, 2, 3], display_labels=SITUATION_LABELS,
    cmap="Blues", ax=ax)
ax.set_title(f"Confusion matrix - {best_name}")
plt.tight_layout()
plt.savefig(ROOT / "outputs" / "classification_confusion_matrix.png", dpi=150)
plt.close()
print("Saved -> outputs/classification_confusion_matrix.png")
