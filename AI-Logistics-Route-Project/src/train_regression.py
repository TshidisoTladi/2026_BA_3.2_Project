"""
train_regression.py - Step 4: Train the REGRESSION model

Goal   : predict traffic volume (Total vehicles per 15 minutes) for a road segment.
Why    : a continuous number = how congested the route will be -> feeds the
         "shortest + safest" route score.
Method : train several regressors, compare them with time-series cross-validation
         on the TRAINING days only, pick the best one, then evaluate it once on the
         untouched TEST days and save it.
"""
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # lets the script save figures without opening a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")  # hide harmless convergence notices

from features import (REGRESSION_FEATURES, REGRESSION_TARGET, build_dataset,
                      chronological_split)
ROOT = Path(__file__).resolve().parent.parent
(ROOT / "models").mkdir(exist_ok=True)
(ROOT / "outputs").mkdir(exist_ok=True)

# ---------------------------------------------------------------- 1. data
data = build_dataset()
train, test = chronological_split(data, test_fraction=0.2)
X_train, y_train = train[REGRESSION_FEATURES], train[REGRESSION_TARGET]
X_test, y_test = test[REGRESSION_FEATURES], test[REGRESSION_TARGET]
print(f"Train: {len(train)} rows ({train['day_index'].nunique()} days) | "
      f"Test: {len(test)} rows ({test['day_index'].nunique()} days)\n")


def report(name, y_true, y_pred):
    return {
        "model": name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


# ------------------------------------------------- 2. baseline to beat
# "Persistence" baseline: assume the next 15 minutes look like the last 15 minutes.
baseline = report("Baseline (previous interval)", y_test, X_test["total_lag1"])

# ------------------------------------------------- 3. candidate models
candidates = {
    "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=42),
    "Gradient Boosting": HistGradientBoostingRegressor(
        learning_rate=0.05, max_iter=300, random_state=42),
    # A small neural network (deep learning family): 2 hidden layers of 64 and 32 neurons
    "Neural Network (MLP)": make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(64, 32), early_stopping=True,
                     max_iter=800, random_state=42)),
}

# Time-series cross-validation: always train on the PAST, validate on the FUTURE.
tscv = TimeSeriesSplit(n_splits=5)
cv_scores = {}
print("Cross-validation on the training days (MAE, lower is better):")
for name, model in candidates.items():
    start = time.time()
    scores = -cross_val_score(model, X_train, y_train, cv=tscv,
                              scoring="neg_mean_absolute_error")
    cv_scores[name] = scores.mean()
    print(f"  {name:<20} CV MAE = {scores.mean():6.2f}  (+/- {scores.std():.2f})"
          f"   [{time.time() - start:.1f}s]")

# ------------------------------------------------- 4. pick + final fit
best_name = min(cv_scores, key=cv_scores.get)
best_model = candidates[best_name]
print(f"\nBest model by cross-validation: {best_name}")
best_model.fit(X_train, y_train)

# ------------------------------------------------- 5. evaluate on TEST days
rows = [baseline]
for name, model in candidates.items():
    if name != best_name:
        model.fit(X_train, y_train)
    rows.append(report(name, y_test, model.predict(X_test)))
results = pd.DataFrame(rows).round(3)
print("\nTest-set results (unseen days):")
print(results.to_string(index=False))
results.to_csv(ROOT / "outputs" / "regression_results.csv", index=False)

# ------------------------------------------------- 6. save model + plots
joblib.dump({"model": best_model, "features": REGRESSION_FEATURES, "name": best_name},
            ROOT / "models" / "regression_model.joblib", compress=3)
print(f"\nSaved -> models/regression_model.joblib")

pred = best_model.predict(X_test)

# Plot A: actual vs predicted for the first 3 test days (time-series view)
n = 96 * 3
plt.figure(figsize=(12, 4))
plt.plot(y_test.values[:n], label="Actual", linewidth=1.5)
plt.plot(pred[:n], label=f"Predicted ({best_name})", linewidth=1.5)
plt.title("Traffic volume: actual vs predicted (first 3 test days)")
plt.xlabel("15-minute intervals")
plt.ylabel("Vehicles per interval")
plt.legend()
plt.tight_layout()
plt.savefig(ROOT / "outputs" / "regression_actual_vs_predicted.png", dpi=150)
plt.close()

# Plot B: which features matter most (permutation importance works for any model)
from sklearn.inspection import permutation_importance
imp = permutation_importance(best_model, X_test, y_test, n_repeats=5,
                             random_state=42, scoring="neg_mean_absolute_error")
imp_df = pd.Series(imp.importances_mean, index=REGRESSION_FEATURES).sort_values()
plt.figure(figsize=(7, 5))
imp_df.plot.barh()
plt.title("Feature importance (increase in MAE when shuffled)")
plt.tight_layout()
plt.savefig(ROOT / "outputs" / "regression_feature_importance.png", dpi=150)
plt.close()
print("Saved -> outputs/regression_actual_vs_predicted.png, regression_feature_importance.png")
print("\nTop 5 features:")
print(imp_df.sort_values(ascending=False).head(5).round(3).to_string())
