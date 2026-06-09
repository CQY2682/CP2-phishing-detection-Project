"""
CP2 Train/Validation/Test Split Analysis
==========================================
Author: Cheah Qi Yang (22095483)

Purpose:
    Demonstrate absence of overfitting using explicit
    train/validation/test three-way split.
    
    If train ≈ validation ≈ test → no overfitting.

Run: python src/train_val_test.py
"""

import os, sys, json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score,
    recall_score, roc_auc_score
)
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FIGURES_DIR = 'reports/figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────
print("Loading data...")
X_all   = pd.read_csv('data/X_train_aug.csv')
y_all   = pd.read_csv('data/y_train_aug.csv')['label']
X_test  = pd.read_csv('data/X_test.csv')
y_test  = pd.read_csv('data/y_test.csv')['label']
print(f"  Full train pool: {X_all.shape}")
print(f"  Test set:        {X_test.shape}")

# ── Split train into train + validation ────────────────────
# 80% of train pool = training, 20% of train pool = validation
# Overall: ~64% train, ~16% val, ~20% test
X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all,
    test_size=0.20,
    stratify=y_all,
    random_state=42
)

print(f"\n  Split summary:")
total = len(X_train) + len(X_val) + len(X_test)
print(f"  Training:   {len(X_train):>7,} ({len(X_train)/total*100:.1f}%)")
print(f"  Validation: {len(X_val):>7,} ({len(X_val)/total*100:.1f}%)")
print(f"  Test:       {len(X_test):>7,} ({len(X_test)/total*100:.1f}%)")
print(f"  Total:      {total:>7,}")


def get_metrics(name, model, X, y):
    pred  = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        'split':     name,
        'accuracy':  round(accuracy_score(y, pred), 4),
        'precision': round(float(precision_score(y, pred)), 4),
        'recall':    round(float(recall_score(y, pred)), 4),
        'f1':        round(float(f1_score(y, pred)), 4),
        'roc_auc':   round(roc_auc_score(y, proba), 4),
    }


# ── XGBoost ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("XGBOOST — Train / Validation / Test")
print("=" * 60)

xgb_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=8, learning_rate=0.1,
    tree_method='hist', n_jobs=-1, random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)

xgb_results = [
    get_metrics('Training',   xgb_model, X_train, y_train),
    get_metrics('Validation', xgb_model, X_val,   y_val),
    get_metrics('Test',       xgb_model, X_test,  y_test),
]

xgb_df = pd.DataFrame(xgb_results).set_index('split')
print(f"\n{xgb_df.to_string()}")

xgb_overfit = abs(xgb_results[0]['f1'] - xgb_results[1]['f1'])
print(f"\n  Train-Validation F1 gap: {xgb_overfit:.4f}")
print(f"  {'No overfitting' if xgb_overfit < 0.01 else 'Possible overfit'} (threshold: 0.01)")


# ── Random Forest ──────────────────────────────────────────
print("\n" + "=" * 60)
print("RANDOM FOREST — Train / Validation / Test")
print("=" * 60)

rf_model = RandomForestClassifier(
    n_estimators=300, max_depth=20,
    class_weight='balanced', n_jobs=-1, random_state=42
)
rf_model.fit(X_train, y_train)

rf_results = [
    get_metrics('Training',   rf_model, X_train, y_train),
    get_metrics('Validation', rf_model, X_val,   y_val),
    get_metrics('Test',       rf_model, X_test,  y_test),
]

rf_df = pd.DataFrame(rf_results).set_index('split')
print(f"\n{rf_df.to_string()}")

rf_overfit = abs(rf_results[0]['f1'] - rf_results[1]['f1'])
print(f"\n  Train-Validation F1 gap: {rf_overfit:.4f}")
print(f"  {'No overfitting' if rf_overfit < 0.01 else 'Possible overfit'} (threshold: 0.01)")


# ── Bar chart comparison ───────────────────────────────────
print("\nGenerating comparison chart...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('#0e1117')

splits  = ['Training', 'Validation', 'Test']
metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
colors  = ['#ff5252', '#ffeb3b', '#00e676']
x       = np.arange(len(metrics))
width   = 0.25

for ax, results, name in zip(axes,
                              [xgb_results, rf_results],
                              ['XGBoost', 'Random Forest']):
    ax.set_facecolor('#0e1117')
    for i, (split, color) in enumerate(zip(splits, colors)):
        vals = [results[i][m] for m in metrics]
        bars = ax.bar(x + i*width, vals, width,
                      label=split, color=color, alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(['Accuracy','Precision','Recall','F1','ROC-AUC'],
                       fontsize=10, color='white')
    ax.set_ylim([0.985, 1.002])
    ax.set_title(f'{name} — Train/Val/Test Comparison',
                 fontsize=12, color='white', pad=10)
    ax.set_ylabel('Score', color='white')
    ax.tick_params(colors='white')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.2)
    ax.spines['bottom'].set_color('#444')
    ax.spines['left'].set_color('#444')

fig.suptitle(
    'Train / Validation / Test Performance\n'
    'Similar scores across all splits confirm no overfitting',
    fontsize=13, color='white', y=1.02
)
plt.tight_layout()
path = os.path.join(FIGURES_DIR, 'train_val_test_comparison.png')
plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0e1117')
plt.close()
print(f"  Saved: {path}")


# ── Save results ───────────────────────────────────────────
output = {
    'split_sizes': {
        'train': len(X_train),
        'validation': len(X_val),
        'test': len(X_test),
        'train_pct': round(len(X_train)/total*100, 1),
        'val_pct': round(len(X_val)/total*100, 1),
        'test_pct': round(len(X_test)/total*100, 1),
    },
    'xgboost': xgb_results,
    'random_forest': rf_results,
    'overfitting_analysis': {
        'xgboost_train_val_gap': round(xgb_overfit, 4),
        'rf_train_val_gap': round(rf_overfit, 4),
        'conclusion': 'No overfitting' if (xgb_overfit < 0.01 and rf_overfit < 0.01)
                      else 'Investigate further'
    }
}

with open('reports/train_val_test_results.json', 'w') as f:
    json.dump(output, f, indent=2)
print(f"  Saved: reports/train_val_test_results.json")

print("\n" + "=" * 60)
print("OVERFITTING ANALYSIS COMPLETE")
print("=" * 60)
print(f"\n  XGBoost  train-val gap: {xgb_overfit:.4f}")
print(f"  RF       train-val gap: {rf_overfit:.4f}")
print(f"\n  Conclusion: {'No overfitting detected' if (xgb_overfit < 0.01 and rf_overfit < 0.01) else 'Investigate'}")
print("\n✅ Done.")