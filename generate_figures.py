"""
Script to reproduce the water potability classification experiment
and generate figures for the ICoICT paper.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, log_loss
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier
from ngboost import NGBClassifier
from ngboost.distns import Bernoulli
from imblearn.combine import SMOTEENN
import os
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

# Output directory for figures
FIG_DIR = '/projects/sandbox/MetodPen-ICoICT-/figures'
os.makedirs(FIG_DIR, exist_ok=True)

# Load dataset
DATA_PATH = '/projects/sandbox/MetodPen-ICoICT-/water_potability.csv'
df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")
print(f"Class distribution:\n{df['Potability'].value_counts()}")

# --- Data Preprocessing ---
X = df.drop('Potability', axis=1)
y = df['Potability']

# Impute missing values with median
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Split: 70% train, 15% val, 15% test
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_imputed, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.176, random_state=42, stratify=y_train_full
)

print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# --- Train Models ---
print("\nTraining NGBoost...")
ngb = NGBClassifier(
    Dist=Bernoulli,
    n_estimators=500,
    learning_rate=0.01,
    random_state=42,
    verbose=False
)
ngb.fit(X_train_scaled, y_train, X_val=X_val_scaled, Y_val=y_val)

print("Training XGBoost...")
xgb = XGBClassifier(
    n_estimators=500,
    learning_rate=0.01,
    max_depth=6,
    random_state=42,
    eval_metric='logloss',
    early_stopping_rounds=50,
    verbosity=0
)
xgb.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)

print("Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_scaled, y_train)

# --- Get Predictions ---
ngb_probs = ngb.predict_proba(X_test_scaled)[:, 1]
ngb_preds = (ngb_probs >= 0.5).astype(int)

xgb_probs = xgb.predict_proba(X_test_scaled)[:, 1]
xgb_preds = (xgb_probs >= 0.5).astype(int)

rf_probs = rf.predict_proba(X_test_scaled)[:, 1]
rf_preds = (rf_probs >= 0.5).astype(int)

# --- Print Metrics ---
def compute_metrics(y_true, y_pred, y_prob, name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    nll = log_loss(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = auc(fpr, tpr)
    print(f"\n{name}:")
    print(f"  Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    print(f"  NLL: {nll:.4f}, AUC: {auc_score:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    return acc, prec, rec, f1, nll, auc_score, cm

ngb_metrics = compute_metrics(y_test, ngb_preds, ngb_probs, "NGBoost")
xgb_metrics = compute_metrics(y_test, xgb_preds, xgb_probs, "XGBoost")
rf_metrics = compute_metrics(y_test, rf_preds, rf_probs, "Random Forest")

# =============================================================================
# FIGURE 1: Confusion Matrices (all 3 models)
# =============================================================================
print("\nGenerating Figure 1: Confusion Matrices...")
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
models_data = [
    ("NGBoost", ngb_metrics[6]),
    ("XGBoost", xgb_metrics[6]),
    ("Random Forest", rf_metrics[6])
]

for idx, (name, cm) in enumerate(models_data):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Not Potable', 'Potable'],
                yticklabels=['Not Potable', 'Potable'])
    axes[idx].set_title(f'{name}')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# FIGURE 2: ROC Curves
# =============================================================================
print("Generating Figure 2: ROC Curves...")
fig, ax = plt.subplots(figsize=(8, 6))

for name, probs, color in [("NGBoost", ngb_probs, 'blue'),
                            ("XGBoost", xgb_probs, 'red'),
                            ("Random Forest", rf_probs, 'green')]:
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc_score = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, label=f'{name} (AUC = {auc_score:.4f})')

ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves Comparison')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'roc_curves.png'), dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# FIGURE 3: Calibration Curves
# =============================================================================
print("Generating Figure 3: Calibration Curves...")
fig, ax = plt.subplots(figsize=(8, 6))

for name, probs, color in [("NGBoost", ngb_probs, 'blue'),
                            ("XGBoost", xgb_probs, 'red'),
                            ("Random Forest", rf_probs, 'green')]:
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test, probs, n_bins=10
    )
    ax.plot(mean_predicted_value, fraction_of_positives, 's-', color=color, label=name)

ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
ax.set_xlabel('Mean Predicted Probability')
ax.set_ylabel('Fraction of Positives')
ax.set_title('Calibration Curves')
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'calibration_curves.png'), dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# FIGURE 4: Probability Distribution (KDE)
# =============================================================================
print("Generating Figure 4: Probability Distributions...")
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for idx, (name, probs) in enumerate([("NGBoost", ngb_probs),
                                      ("XGBoost", xgb_probs),
                                      ("Random Forest", rf_probs)]):
    class_0_probs = probs[y_test == 0]
    class_1_probs = probs[y_test == 1]
    sns.kdeplot(class_0_probs, ax=axes[idx], label='Not Potable (0)', color='blue', fill=True, alpha=0.3)
    sns.kdeplot(class_1_probs, ax=axes[idx], label='Potable (1)', color='red', fill=True, alpha=0.3)
    axes[idx].set_title(f'{name}')
    axes[idx].set_xlabel('Predicted Probability')
    axes[idx].set_ylabel('Density')
    axes[idx].legend()

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'probability_distributions.png'), dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# FIGURE 5: Feature Importance
# =============================================================================
print("Generating Figure 5: Feature Importance...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# XGBoost feature importance
xgb_importance = xgb.feature_importances_
xgb_fi = pd.Series(xgb_importance, index=X.columns).sort_values(ascending=True)
xgb_fi.plot(kind='barh', ax=axes[0], color='indianred')
axes[0].set_title('XGBoost')
axes[0].set_xlabel('Feature Importance')

# Random Forest feature importance
rf_importance = rf.feature_importances_
rf_fi = pd.Series(rf_importance, index=X.columns).sort_values(ascending=True)
rf_fi.plot(kind='barh', ax=axes[1], color='forestgreen')
axes[1].set_title('Random Forest')
axes[1].set_xlabel('Feature Importance')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'feature_importance.png'), dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# FIGURE 6: XGBoost Validation Loss Curve
# =============================================================================
print("Generating Figure 6: XGBoost Loss Curve...")
fig, ax = plt.subplots(figsize=(8, 5))

results = xgb.evals_result()
val_loss = results['validation_0']['logloss']
ax.plot(range(len(val_loss)), val_loss, 'b-', label='Validation Loss')
ax.set_xlabel('Boosting Rounds')
ax.set_ylabel('Log Loss')
ax.set_title('XGBoost Validation Loss Curve')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'xgboost_loss_curve.png'), dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# FIGURE 7: SMOTE-ENN Comparison
# =============================================================================
print("Generating Figure 7: SMOTE-ENN Comparison...")

# Apply SMOTE-ENN
smote_enn = SMOTEENN(random_state=42)
X_resampled, y_resampled = smote_enn.fit_resample(X_train_scaled, y_train)
print(f"  Before SMOTE-ENN: {len(X_train_scaled)} samples")
print(f"  After SMOTE-ENN: {len(X_resampled)} samples")

# Retrain with SMOTE-ENN data
ngb_smote = NGBClassifier(Dist=Bernoulli, n_estimators=500, learning_rate=0.01, random_state=42, verbose=False)
ngb_smote.fit(X_resampled, y_resampled, X_val=X_val_scaled, Y_val=y_val)

xgb_smote = XGBClassifier(n_estimators=500, learning_rate=0.01, max_depth=6, random_state=42,
                           eval_metric='logloss', early_stopping_rounds=50, verbosity=0)
xgb_smote.fit(X_resampled, y_resampled, eval_set=[(X_val_scaled, y_val)], verbose=False)

rf_smote = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
rf_smote.fit(X_resampled, y_resampled)

# Compute accuracies with SMOTE-ENN
ngb_smote_acc = accuracy_score(y_test, ngb_smote.predict(X_test_scaled))
xgb_smote_acc = accuracy_score(y_test, xgb_smote.predict(X_test_scaled))
rf_smote_acc = accuracy_score(y_test, rf_smote.predict(X_test_scaled))

print(f"  NGBoost acc without/with SMOTE-ENN: {ngb_metrics[0]:.4f} / {ngb_smote_acc:.4f}")
print(f"  XGBoost acc without/with SMOTE-ENN: {xgb_metrics[0]:.4f} / {xgb_smote_acc:.4f}")
print(f"  RF acc without/with SMOTE-ENN: {rf_metrics[0]:.4f} / {rf_smote_acc:.4f}")

# Bar plot comparison
fig, ax = plt.subplots(figsize=(8, 5))
models = ['NGBoost', 'XGBoost', 'Random Forest']
without_smote = [ngb_metrics[0], xgb_metrics[0], rf_metrics[0]]
with_smote = [ngb_smote_acc, xgb_smote_acc, rf_smote_acc]

x = np.arange(len(models))
width = 0.35
bars1 = ax.bar(x - width/2, without_smote, width, label='Without SMOTE-ENN', color='steelblue')
bars2 = ax.bar(x + width/2, with_smote, width, label='With SMOTE-ENN', color='coral')

ax.set_ylabel('Accuracy')
ax.set_title('Impact of SMOTE-ENN on Model Accuracy')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.set_ylim(0.4, 0.8)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'smote_enn_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# FIGURE 8: Uncertainty Zone Analysis
# =============================================================================
print("Generating Figure 8: Uncertainty Zone Analysis...")
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

def uncertainty_zone_analysis(probs, y_true, ax, title):
    zones = [
        ("Zone 1\n(< 0.2)", probs < 0.2),
        ("Zone 2\n(0.2-0.4)", (probs >= 0.2) & (probs < 0.4)),
        ("Zone 3\n(0.4-0.6)", (probs >= 0.4) & (probs < 0.6)),
        ("Zone 4\n(0.6-0.8)", (probs >= 0.6) & (probs < 0.8)),
        ("Zone 5\n(>= 0.8)", probs >= 0.8),
    ]
    zone_names = []
    zone_accs = []
    zone_counts = []
    
    for name, mask in zones:
        if mask.sum() > 0:
            preds = (probs[mask] >= 0.5).astype(int)
            acc = accuracy_score(y_true[mask], preds)
            zone_names.append(name)
            zone_accs.append(acc)
            zone_counts.append(mask.sum())
        else:
            zone_names.append(name)
            zone_accs.append(0)
            zone_counts.append(0)
    
    colors = ['#2ecc71', '#f1c40f', '#e74c3c', '#f1c40f', '#2ecc71']
    bars = ax.bar(zone_names, zone_accs, color=colors, edgecolor='black', alpha=0.7)
    ax.set_title(title)
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.1)
    
    for bar, count in zip(bars, zone_counts):
        height = bar.get_height()
        ax.annotate(f'n={count}\n{height:.3f}', 
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

uncertainty_zone_analysis(ngb_probs, y_test.values, axes[0], "NGBoost")
uncertainty_zone_analysis(xgb_probs, y_test.values, axes[1], "XGBoost")
uncertainty_zone_analysis(rf_probs, y_test.values, axes[2], "Random Forest")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'uncertainty_zones.png'), dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# FIGURE 9: Missing Values
# =============================================================================
print("Generating Figure 9: Missing Values...")
fig, ax = plt.subplots(figsize=(8, 4))
missing_pct = (df.isnull().sum() / len(df)) * 100
missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
missing_pct.plot(kind='bar', ax=ax, color='coral', edgecolor='black')
ax.set_ylabel('Missing Values (%)')
ax.set_title('Percentage of Missing Values per Feature')
ax.set_xlabel('')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'missing_values.png'), dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# FIGURE 10: Class Distribution
# =============================================================================
print("Generating Figure 10: Class Distribution...")
fig, ax = plt.subplots(figsize=(6, 4))
class_counts = df['Potability'].value_counts()
colors = ['steelblue', 'coral']
bars = ax.bar(['Not Potable (0)', 'Potable (1)'], class_counts.values, color=colors, edgecolor='black')
ax.set_ylabel('Count')
ax.set_title('Class Distribution in Water Potability Dataset')
for bar, count in zip(bars, class_counts.values):
    pct = count / len(df) * 100
    ax.annotate(f'{count}\n({pct:.1f}%)', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 5), textcoords="offset points", ha='center', va='bottom')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'class_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()

print("\n=== All figures generated successfully! ===")
print(f"Figures saved to: {FIG_DIR}")
