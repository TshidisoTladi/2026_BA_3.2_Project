"""
evaluate_models.py - Step 4: evaluate the SAVED models in depth on the unseen test days

Goes beyond one accuracy number and shows WHERE the models are strong or weak:
  Regression     : MAE / RMSE / R2, MAE as a % of average volume, residual plot,
                   error by hour of day, error under different weather / hazards
  Classification : accuracy, macro-F1, per-class report, confusion matrix
Everything is saved to outputs/ so it can go straight into your report and poster.
"""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score)

from features import (CLASSIFICATION_TARGET, REGRESSION_TARGET, build_dataset,
                      chronological_split)

ROOT = Path(__file__).resolve().parent.parent
_, test = chronological_split(build_dataset())
test = test.copy()

# ================================================================ REGRESSION
reg = joblib.load(ROOT / "models" / "regression_model.joblib")
y = test[REGRESSION_TARGET]
test["pred"] = reg["model"].predict(test[reg["features"]])
test["error"] = test["pred"] - y
test["abs_error"] = test["error"].abs()

mae = mean_absolute_error(y, test["pred"])
rmse = np.sqrt(mean_squared_error(y, test["pred"]))
print("=" * 60, f"\nREGRESSION - {reg['name']}\n" + "=" * 60)
print(f"MAE   : {mae:.2f} vehicles per 15 min")
print(f"RMSE  : {rmse:.2f}")
print(f"R2    : {r2_score(y, test['pred']):.3f}")
print(f"MAE as % of average volume ({y.mean():.0f}): {mae / y.mean() * 100:.1f}%")
print(f"Predictions within +/-20 vehicles: {(test['abs_error'] <= 20).mean() * 100:.1f}%")
print(f"Bias (mean error; + means over-predicting): {test['error'].mean():+.2f}")

print("\nMAE by traffic situation (where is the model weakest?):")
print(test.groupby("traffic_situation")["abs_error"].mean().round(1).to_string())
print("\nMAE by weather severity (0 = fine ... 3 = severe):")
print(test.groupby("weather_severity")["abs_error"].agg(["mean", "count"]).round(1).to_string())
print("\nMAE with / without an accident:", test.groupby("accidents")["abs_error"].mean().round(1).to_dict())
print("MAE road open / closed        :", test.groupby("road_closed")["abs_error"].mean().round(1).to_dict())

fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
axes[0].scatter(y, test["pred"], s=8, alpha=0.4)
lim = [0, max(y.max(), test["pred"].max())]
axes[0].plot(lim, lim, "r--", label="Perfect prediction")
axes[0].set_title("Actual vs predicted")
axes[0].set_xlabel("Actual volume")
axes[0].set_ylabel("Predicted volume")
axes[0].legend()

axes[1].hist(test["error"], bins=40)
axes[1].axvline(0, color="red", linestyle="--")
axes[1].set_title("Residuals (predicted - actual)")
axes[1].set_xlabel("Error in vehicles")

test.groupby("hour")["abs_error"].mean().plot.bar(ax=axes[2], rot=90)
axes[2].set_title("Average error by hour of day")
axes[2].set_xlabel("Hour")
axes[2].set_ylabel("MAE")
plt.tight_layout()
plt.savefig(ROOT / "outputs" / "eval_regression.png", dpi=150)
plt.close()

# ============================================================ CLASSIFICATION
clf = joblib.load(ROOT / "models" / "classification_model.joblib")
yc = test[CLASSIFICATION_TARGET].astype(int)
pc = clf["model"].predict(test[clf["features"]])
print("\n" + "=" * 60, f"\nCLASSIFICATION - {clf['name']}\n" + "=" * 60)
print(f"Accuracy : {accuracy_score(yc, pc):.3f}")
print(f"Macro-F1 : {f1_score(yc, pc, average='macro'):.3f}")
print(classification_report(yc, pc, labels=[0, 1, 2, 3], target_names=clf["labels"],
                            zero_division=0))

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(yc, pc, labels=[0, 1, 2, 3],
                                        display_labels=clf["labels"], cmap="Blues", ax=ax)
ax.set_title(f"Confusion matrix - {clf['name']}")
plt.tight_layout()
plt.savefig(ROOT / "outputs" / "eval_classification.png", dpi=150)
plt.close()
print("Saved -> outputs/eval_regression.png, outputs/eval_classification.png")
