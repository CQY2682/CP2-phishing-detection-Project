"""
CP2 Learning Curve Analysis
============================
Author: Cheah Qi Yang (22095483)

Purpose:
    Demonstrate absence of overfitting by plotting training vs
    validation score across increasing training set sizes.
    
    A converging gap between train and validation curves
    confirms the model generalises well (no overfitting).

Run: python src/learning_curve.py
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import learning_curve
from sklearn.metrics import f1_score, make_scorer
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FIGURES_DIR = 'reports/figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────
print("Loading training data...")
X_train = pd.read_csv('data/X_train_aug.csv')
y_train = pd.read_csv('data/y_train_aug.csv')['label']
X_test  = pd.read_csv('data/X_test.csv')
y_test  = pd.read_csv('data/y_test.csv')['label']
print(f"  Train: {X_train.shape} | Test: {X_test.shape}")

f1_scorer = make_scorer(f1_score)

# ── Learning curve function ────────────────────────────────
def plot_learning_curve(estimator, name, X, y, filename):
    print(f"\nGenerating learning curve for {name}...")
    print("  This takes a few minutes...")

    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y,
        cv=5,
        scoring='f1',
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1,
        verbose=0
    )

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    # Plot
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    # Training score
    ax.plot(train_sizes, train_mean, 'o-', color='#ff5252',
            linewidth=2, label='Training Score', markersize=6)
    ax.fill_between(train_sizes,
                    train_mean - train_std,
                    train_mean + train_std,
                    alpha=0.15, color='#ff5252')

    # Validation score
    ax.plot(train_sizes, val_mean, 'o-', color='#00e676',
            linewidth=2, label='Cross-Validation Score', markersize=6)
    ax.fill_between(train_sizes,
                    val_mean - val_std,
                    val_mean + val_std,
                    alpha=0.15, color='#00e676')

    ax.set_xlabel('Training Set Size', fontsize=13, color='white')
    ax.set_ylabel('F1 Score', fontsize=13, color='white')
    ax.set_title(f'Learning Curve — {name}\n(Converging curves = no overfitting)',
                 fontsize=13, color='white', pad=12)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.2)
    ax.tick_params(colors='white')
    ax.set_ylim([0.93, 1.01])

    # Add gap annotation at final point
    final_gap = abs(train_mean[-1] - val_mean[-1])
    ax.annotate(f'Final gap: {final_gap:.4f}',
                xy=(train_sizes[-1], val_mean[-1]),
                xytext=(train_sizes[-1] * 0.7, val_mean[-1] - 0.01),
                fontsize=10, color='#ffeb3b',
                arrowprops=dict(arrowstyle='->', color='#ffeb3b'))

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight',
                facecolor='#0e1117')
    plt.close()
    print(f"  Saved: {path}")

    # Print summary
    print(f"\n  {name} Learning Curve Summary:")
    print(f"  {'Train Size':>12} {'Train F1':>10} {'Val F1':>10} {'Gap':>8}")
    print(f"  {'-'*44}")
    for i, size in enumerate(train_sizes):
        gap = abs(train_mean[i] - val_mean[i])
        print(f"  {int(size):>12,} {train_mean[i]:>10.4f} {val_mean[i]:>10.4f} {gap:>8.4f}")

    return train_mean, val_mean, train_sizes


# ── XGBoost learning curve ─────────────────────────────────
xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    tree_method='hist',
    n_jobs=-1,
    random_state=42,
    eval_metric='logloss'
)

xgb_train, xgb_val, sizes = plot_learning_curve(
    xgb_model, "XGBoost",
    X_train, y_train,
    "learning_curve_xgboost.png"
)

# ── Random Forest learning curve ───────────────────────────
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)

rf_train, rf_val, _ = plot_learning_curve(
    rf_model, "Random Forest",
    X_train, y_train,
    "learning_curve_rf.png"
)

# ── Combined plot ──────────────────────────────────────────
print("\nGenerating combined comparison plot...")
plt.style.use('dark_background')
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor('#0e1117')

for ax, train_m, val_m, name, color in zip(
    axes,
    [xgb_train, rf_train],
    [xgb_val, rf_val],
    ['XGBoost', 'Random Forest'],
    ['#ff5252', '#2196f3']
):
    ax.set_facecolor('#0e1117')
    ax.plot(sizes, train_m, 'o-', color=color,
            linewidth=2, label='Training', markersize=5)
    ax.plot(sizes, val_m, 'o--', color='#00e676',
            linewidth=2, label='Validation', markersize=5)
    ax.fill_between(sizes, train_m, val_m, alpha=0.1, color='yellow')
    ax.set_title(f'{name}', fontsize=13, color='white')
    ax.set_xlabel('Training Size', fontsize=11, color='white')
    ax.set_ylabel('F1 Score', fontsize=11, color='white')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.tick_params(colors='white')
    ax.set_ylim([0.93, 1.01])

fig.suptitle(
    'Learning Curves — Training vs Validation F1\n'
    'Converging lines confirm absence of overfitting',
    fontsize=14, color='white', y=1.02
)
plt.tight_layout()
path = os.path.join(FIGURES_DIR, 'learning_curves_combined.png')
plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0e1117')
plt.close()
print(f"  Saved: {path}")

# ── Final verdict ──────────────────────────────────────────
print("\n" + "=" * 60)
print("OVERFITTING ANALYSIS COMPLETE")
print("=" * 60)
xgb_gap = abs(xgb_train[-1] - xgb_val[-1])
rf_gap  = abs(rf_train[-1] - rf_val[-1])
print(f"\n  XGBoost  final gap (train - val): {xgb_gap:.4f}")
print(f"  RF       final gap (train - val): {rf_gap:.4f}")

if xgb_gap < 0.01 and rf_gap < 0.01:
    print("\n  CONCLUSION: No overfitting detected.")
    print("  Both models show converging train/validation curves.")
    print("  Gap < 0.01 confirms strong generalization.")
else:
    print(f"\n  WARNING: Gap > 0.01 detected. Investigate further.")

print(f"\n  Figures saved to: {FIGURES_DIR}/")
print("\n✅ Learning curve analysis complete.")