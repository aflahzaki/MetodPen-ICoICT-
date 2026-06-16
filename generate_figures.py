"""
Faithful reproduction of the BAB3_MetodPen_Real.ipynb (Colab) experiment.

This script replicates the EXACT pipeline used in the source notebook so that the
figures generated here are consistent with the numbers reported in the paper tables:
    - MICE (IterativeImputer) for missing-value imputation
    - 70/15/15 stratified split (test_size=0.15, then val test_size=15/85)
    - StandardScaler fit on the training split only
    - SMOTE-ENN evaluated but NOT applied to the final models (it degrades performance)
    - NGBoost : 300 estimators, lr=0.05, minibatch_frac=0.8, col_sample=0.8
    - XGBoost : 300 estimators, lr=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8
    - RandomForest : 300 trees

Run:
    python3 generate_figures.py
"""
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, roc_auc_score, roc_curve, confusion_matrix,
)
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from ngboost import NGBClassifier
from ngboost.distns import Bernoulli
from imblearn.combine import SMOTEENN
from scipy.stats import chi2

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("paper", font_scale=1.2)

RANDOM_STATE = 42
BASE = "/projects/sandbox/MetodPen-ICoICT-"
FIG_DIR = os.path.join(BASE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
COLORS = {"NGBoost": "#2196F3", "XGBoost": "#4CAF50", "Random Forest": "#FF9800"}

# --------------------------------------------------------------------------
# 1. Load data
# --------------------------------------------------------------------------
df = pd.read_csv(os.path.join(BASE, "water_potability.csv"))
class_dist = df["Potability"].value_counts()
print(f"Dataset shape: {df.shape}")
print(f"Class 0: {class_dist[0]} ({class_dist[0] / len(df) * 100:.2f}%), "
      f"Class 1: {class_dist[1]} ({class_dist[1] / len(df) * 100:.2f}%)")

X = df.drop("Potability", axis=1)
y = df["Potability"]

# --------------------------------------------------------------------------
# 2. MICE imputation (matches Colab IterativeImputer)
# --------------------------------------------------------------------------
mice = IterativeImputer(max_iter=10, random_state=RANDOM_STATE, sample_posterior=False)
X_imputed = pd.DataFrame(mice.fit_transform(X), columns=X.columns)

# --------------------------------------------------------------------------
# 3. 70/15/15 stratified split (exact Colab fractions)
# --------------------------------------------------------------------------
X_temp, X_test, y_temp, y_test = train_test_split(
    X_imputed, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=15 / 85, stratify=y_temp, random_state=RANDOM_STATE)
print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------------------------------
# 4. SMOTE-ENN (evaluated, NOT applied to final models)
# --------------------------------------------------------------------------
X_train_resampled, y_train_resampled = SMOTEENN(
    random_state=RANDOM_STATE).fit_resample(X_train_scaled, y_train)
print(f"SMOTE-ENN: {len(X_train_scaled)} -> {len(X_train_resampled)} samples")

# Final models train on the ORIGINAL (non-resampled) scaled training data
X_train_final, y_train_final = X_train_scaled, y_train


def build_models():
    ngb = NGBClassifier(Dist=Bernoulli, n_estimators=300, learning_rate=0.05,
                        minibatch_frac=0.8, col_sample=0.8,
                        random_state=RANDOM_STATE, verbose=False)
    xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                        subsample=0.8, colsample_bytree=0.8,
                        random_state=RANDOM_STATE, eval_metric="logloss", verbosity=0)
    rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    return ngb, xgb, rf


# --------------------------------------------------------------------------
# 5. Train final models
# --------------------------------------------------------------------------
print("Training final models (NGBoost, XGBoost, Random Forest)...")
ngb_model, xgb_model, rf_model = build_models()
np.random.seed(RANDOM_STATE)  # ensure NGBoost minibatch/col subsampling is reproducible
ngb_model.fit(X_train_final, y_train_final, X_val=X_val_scaled, Y_val=y_val)
xgb_model.fit(X_train_final, y_train_final, eval_set=[(X_val_scaled, y_val)], verbose=False)
rf_model.fit(X_train_final, y_train_final)

models = {"NGBoost": ngb_model, "XGBoost": xgb_model, "Random Forest": rf_model}


# --------------------------------------------------------------------------
# 6. Evaluate (metrics + ECE, matching Colab exactly)
# --------------------------------------------------------------------------
def expected_calibration_error(y_true, y_prob, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    ece, n_total = 0.0, len(y_true)
    yt = np.asarray(y_true)
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (y_prob >= edges[i]) & (y_prob <= edges[i + 1])
        else:
            mask = (y_prob >= edges[i]) & (y_prob < edges[i + 1])
        if mask.sum() > 0:
            ece += (mask.sum() / n_total) * abs(yt[mask].mean() - y_prob[mask].mean())
    return ece


results = {}
for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)
    y_prob_pos = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    results[name] = {
        "y_pred": y_pred, "y_prob_pos": y_prob_pos, "cm": cm,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "nll": log_loss(y_test, y_prob_pos),
        "ece": expected_calibration_error(y_test, y_prob_pos),
        "auc": roc_auc_score(y_test, y_prob_pos),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }

print("\n=== FINAL METRICS (Test Set, N={}) ===".format(len(y_test)))
print(f"{'Metric':<12}{'NGBoost':<12}{'XGBoost':<12}{'Random Forest':<14}")
for m in ["accuracy", "precision", "recall", "f1", "nll", "ece", "auc"]:
    print(f"{m:<12}{results['NGBoost'][m]:<12.4f}{results['XGBoost'][m]:<12.4f}"
          f"{results['Random Forest'][m]:<14.4f}")
print("\nConfusion (TN,FP,FN,TP):")
for name in models:
    r = results[name]
    print(f"  {name:<14} TN={r['tn']}, FP={r['fp']}, FN={r['fn']}, TP={r['tp']}")


# --------------------------------------------------------------------------
# 7. McNemar's test
# --------------------------------------------------------------------------
def mcnemar(y_true, pa, pb):
    ca, cb = (pa == y_true), (pb == y_true)
    n01 = ((~ca) & cb).sum()
    n10 = (ca & (~cb)).sum()
    if (n01 + n10) == 0:
        return 0.0, 1.0
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    return stat, 1 - chi2.cdf(stat, df=1)


print("\n=== McNEMAR ===")
names = list(models.keys())
yt = y_test.values
mcnemar_list = []
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        s, p = mcnemar(yt, results[names[i]]["y_pred"], results[names[j]]["y_pred"])
        print(f"  {names[i]} vs {names[j]}: chi2={s:.4f}, p={p:.4f}")
        mcnemar_list.append({"pair": f"{names[i]} vs {names[j]}",
                             "chi2": round(float(s), 4), "p": round(float(p), 4),
                             "significant": bool(p < 0.05)})


# --------------------------------------------------------------------------
# 8. Uncertainty zones
# --------------------------------------------------------------------------
ZONES = [("Zone 1", "mu < 0.2", 0.0, 0.2), ("Zone 2", "0.2 - 0.4", 0.2, 0.4),
         ("Zone 3", "0.4 - 0.6", 0.4, 0.6), ("Zone 4", "0.6 - 0.8", 0.6, 0.8),
         ("Zone 5", "mu >= 0.8", 0.8, 1.01)]
print("\n=== UNCERTAINTY ZONES ===")
zones_dict = {}
for name in models:
    probs, preds = results[name]["y_prob_pos"], results[name]["y_pred"]
    print(f"  {name}:")
    zones_dict[name] = []
    for zlabel, zrange, lo, hi in ZONES:
        mask = (probs >= lo) & (probs < hi)
        n = int(mask.sum())
        acc = float(accuracy_score(yt[mask], preds[mask])) if n > 0 else 0.0
        avgp = float(probs[mask].mean()) if n > 0 else 0.0
        print(f"    {zlabel} ({zrange}) N={n:<4} Acc={acc:.4f} AvgP={avgp:.4f}")
        zones_dict[name].append({"zone": zlabel, "range": zrange, "n": n,
                                 "acc": round(acc, 4), "avg_prob": round(avgp, 4)})


# --------------------------------------------------------------------------
# 9. SMOTE-ENN comparison (train on resampled, evaluate on test)
# --------------------------------------------------------------------------
print("\nTraining SMOTE-ENN models for comparison...")
ngb_s, xgb_s, rf_s = build_models()
np.random.seed(RANDOM_STATE)  # reproducible NGBoost subsampling
ngb_s.fit(X_train_resampled, y_train_resampled)
xgb_s.fit(X_train_resampled, y_train_resampled)
rf_s.fit(X_train_resampled, y_train_resampled)
smote_acc = {
    "NGBoost": accuracy_score(y_test, ngb_s.predict(X_test_scaled)),
    "XGBoost": accuracy_score(y_test, xgb_s.predict(X_test_scaled)),
    "Random Forest": accuracy_score(y_test, rf_s.predict(X_test_scaled)),
}
for name in models:
    print(f"  {name}: {results[name]['accuracy']:.4f} (no) vs {smote_acc[name]:.4f} (smote)")

# ==========================================================================
# FIGURES
# ==========================================================================
print("\nGenerating figures...")

# Fig 1: class distribution
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(["Not Potable (0)", "Potable (1)"], [class_dist[0], class_dist[1]],
              color=["steelblue", "coral"], edgecolor="black")
ax.set_ylabel("Count")
ax.set_title("Class Distribution in Water Potability Dataset")
for b, c in zip(bars, [class_dist[0], class_dist[1]]):
    ax.annotate(f"{c}\n({c/len(df)*100:.1f}%)",
                xy=(b.get_x() + b.get_width()/2, b.get_height()),
                xytext=(0, 5), textcoords="offset points", ha="center")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "class_distribution.png"), dpi=200, bbox_inches="tight")
plt.close()

# Fig 2: missing values
fig, ax = plt.subplots(figsize=(8, 4))
mv = (df.isnull().sum() / len(df) * 100)
mv = mv[mv > 0].sort_values(ascending=False)
mv.plot(kind="bar", ax=ax, color="coral", edgecolor="black")
ax.set_ylabel("Missing Values (%)")
ax.set_title("Percentage of Missing Values per Feature")
ax.set_xlabel("")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "missing_values.png"), dpi=200, bbox_inches="tight")
plt.close()

# Fig 3: confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, name in zip(axes, models):
    sns.heatmap(results[name]["cm"], annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Not Potable", "Potable"],
                yticklabels=["Not Potable", "Potable"])
    ax.set_title(f"{name}\nAcc={results[name]['accuracy']:.4f}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "confusion_matrices.png"), dpi=200, bbox_inches="tight")
plt.close()

# Fig 4: calibration curves
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
for name in models:
    ft, fp_ = calibration_curve(y_test, results[name]["y_prob_pos"], n_bins=10, strategy="uniform")
    ax.plot(fp_, ft, "o-", color=COLORS[name], label=f"{name} (ECE={results[name]['ece']:.4f})")
ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Fraction of Positives")
ax.set_title("Calibration Curves (Reliability Diagram)")
ax.legend(loc="lower right")
ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "calibration_curves.png"), dpi=200, bbox_inches="tight")
plt.close()

# Fig 5: ROC curves
fig, ax = plt.subplots(figsize=(8, 6))
for name in models:
    fpr, tpr, _ = roc_curve(y_test, results[name]["y_prob_pos"])
    ax.plot(fpr, tpr, color=COLORS[name], label=f"{name} (AUC={results[name]['auc']:.4f})")
ax.plot([0, 1], [0, 1], "k--", label="Random Classifier")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "roc_curves.png"), dpi=200, bbox_inches="tight")
plt.close()

# Fig 6: uncertainty zones
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
zone_labels = ["Zone 1\n(<0.2)", "Zone 2\n(0.2-0.4)", "Zone 3\n(0.4-0.6)",
               "Zone 4\n(0.6-0.8)", "Zone 5\n(>=0.8)"]
zone_colors = ["#2ecc71", "#f1c40f", "#e74c3c", "#f1c40f", "#2ecc71"]
for ax, name in zip(axes, models):
    probs, preds = results[name]["y_prob_pos"], results[name]["y_pred"]
    accs, counts = [], []
    for _zl, _zr, lo, hi in ZONES:
        mask = (probs >= lo) & (probs < hi)
        counts.append(int(mask.sum()))
        accs.append(accuracy_score(yt[mask], preds[mask]) if mask.sum() > 0 else 0)
    bars = ax.bar(zone_labels, accs, color=zone_colors, edgecolor="black", alpha=0.75)
    ax.set_title(name); ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.15)
    for b, c in zip(bars, counts):
        ax.annotate(f"n={c}\n{b.get_height():.3f}",
                    xy=(b.get_x() + b.get_width()/2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "uncertainty_zones.png"), dpi=200, bbox_inches="tight")
plt.close()

# Fig 7: KDE probability distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, name in zip(axes, models):
    probs = results[name]["y_prob_pos"]
    sns.kdeplot(probs[(y_test == 0).values], ax=ax, color="red",
                label="Not Potable (0)", fill=True, alpha=0.3)
    sns.kdeplot(probs[(y_test == 1).values], ax=ax, color="blue",
                label="Potable (1)", fill=True, alpha=0.3)
    ax.axvline(0.5, color="black", linestyle="--", alpha=0.5)
    ax.set_title(name); ax.set_xlabel("Predicted P(Potable)")
    ax.set_ylabel("Density"); ax.set_xlim([0, 1]); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "probability_distributions.png"), dpi=200, bbox_inches="tight")
plt.close()

# Fig 8: SMOTE-ENN comparison
fig, ax = plt.subplots(figsize=(8, 5))
xp = np.arange(3); w = 0.35
no_s = [results[n]["accuracy"] for n in ["NGBoost", "XGBoost", "Random Forest"]]
ys = [smote_acc[n] for n in ["NGBoost", "XGBoost", "Random Forest"]]
b1 = ax.bar(xp - w/2, no_s, w, label="Without SMOTE-ENN", color="steelblue")
b2 = ax.bar(xp + w/2, ys, w, label="With SMOTE-ENN", color="coral")
ax.set_ylabel("Accuracy"); ax.set_title("Impact of SMOTE-ENN on Model Accuracy")
ax.set_xticks(xp); ax.set_xticklabels(["NGBoost", "XGBoost", "Random Forest"])
ax.legend(); ax.set_ylim(0.4, 0.8)
for bars in (b1, b2):
    for b in bars:
        ax.annotate(f"{b.get_height():.4f}",
                    xy=(b.get_x() + b.get_width()/2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "smote_enn_comparison.png"), dpi=200, bbox_inches="tight")
plt.close()

# Fig 9: XGBoost validation loss curve
fig, ax = plt.subplots(figsize=(8, 5))
evals = xgb_model.evals_result()
if evals and "validation_0" in evals:
    vl = evals["validation_0"]["logloss"]
    ax.plot(range(len(vl)), vl, color=COLORS["XGBoost"], label="Validation Loss")
ax.set_xlabel("Boosting Rounds"); ax.set_ylabel("Log Loss")
ax.set_title("XGBoost Validation Loss Curve"); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "xgboost_loss_curve.png"), dpi=200, bbox_inches="tight")
plt.close()

# Fig 10: feature importance (XGBoost and Random Forest only).
# NGBoost's tree-level importances do not have a single well-defined aggregation
# for distributional parameters, so we report the two tree-ensemble baselines.
feat = X.columns.tolist()
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, imp, name in zip(
        axes, [xgb_model.feature_importances_, rf_model.feature_importances_],
        ["XGBoost", "Random Forest"]):
    order = np.argsort(imp)
    ax.barh(range(len(feat)), np.asarray(imp)[order], color=COLORS[name])
    ax.set_yticks(range(len(feat)))
    ax.set_yticklabels([feat[i] for i in order])
    ax.set_xlabel("Importance"); ax.set_title(name)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "feature_importance.png"), dpi=200, bbox_inches="tight")
plt.close()

print("\nAll figures regenerated to match the Colab pipeline.")

# ==========================================================================
# EXPORT results.json  (single source of truth for the paper)
# ==========================================================================
import json
import sklearn
import xgboost
import ngboost as _ngb

feat = X.columns.tolist()
fi = {
    "XGBoost": sorted(zip(feat, [float(v) for v in xgb_model.feature_importances_]), key=lambda t: -t[1]),
    "Random Forest": sorted(zip(feat, [float(v) for v in rf_model.feature_importances_]), key=lambda t: -t[1]),
}

summary = {
    "dataset": {
        "n_samples": int(len(df)), "n_features": int(X.shape[1]),
        "class0": int(class_dist[0]), "class1": int(class_dist[1]),
        "pct0": round(class_dist[0] / len(df) * 100, 2),
        "pct1": round(class_dist[1] / len(df) * 100, 2),
    },
    "missing_pct": {k: round(float(v), 2)
                    for k, v in ((df.isnull().sum() / len(df) * 100).items()) if v > 0},
    "split": {"train": int(len(X_train)), "val": int(len(X_val)), "test": int(len(X_test))},
    "smote": {"before": int(len(X_train_scaled)), "after": int(len(X_train_resampled))},
    "metrics": {name: {m: round(float(results[name][m]), 4)
                       for m in ["accuracy", "precision", "recall", "f1", "nll", "ece", "auc"]}
                for name in models},
    "confusion": {name: {k: int(results[name][k]) for k in ["tn", "fp", "fn", "tp"]}
                  for name in models},
    "mcnemar": mcnemar_list,
    "zones": zones_dict,
    "smote_acc": {name: {"without": round(float(results[name]["accuracy"]), 4),
                         "with": round(float(smote_acc[name]), 4),
                         "diff": round(float(smote_acc[name] - results[name]["accuracy"]), 4)}
                  for name in models},
    "feature_importance": {name: [[f, round(v, 4)] for f, v in fi[name]] for name in fi},
    "environment": {
        "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__,
        "ngboost": _ngb.__version__, "random_state": RANDOM_STATE,
    },
}

with open(os.path.join(BASE, "results.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("results.json written (single source of truth for the paper).")
