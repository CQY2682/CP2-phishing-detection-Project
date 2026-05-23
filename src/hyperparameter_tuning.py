"""
CP2 Hyperparameter Tuning
==========================
Author: Cheah Qi Yang (22095483)

Purpose:
    Validate that current hyperparameters are optimal using
    5-fold cross-validation grid search.
    
    Justifies parameter choices in thesis methodology section.

Run: python src/hyperparameter_tuning.py
Note: Takes 15-40 minutes depending on grid size.
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import f1_score, make_scorer
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Load data ──────────────────────────────────────────────
print("Loading training data...")
X_train = pd.read_csv('data/X_train_aug.csv')
y_train = pd.read_csv('data/y_train_aug.csv')['label']
print(f"  Shape: {X_train.shape}")

f1_scorer = make_scorer(f1_score)

# ── XGBoost Grid Search ────────────────────────────────────
print("\n" + "=" * 60)
print("XGBOOST HYPERPARAMETER TUNING")
print("=" * 60)
print("Testing parameter combinations (5-fold CV)...")
print("This may take 10-20 minutes...\n")

xgb_params = {
    'n_estimators':  [100, 200, 300, 500],
    'max_depth':     [4, 6, 8, 10],
    'learning_rate': [0.05, 0.1, 0.2],
}

xgb_base = xgb.XGBClassifier(
    tree_method='hist',
    n_jobs=-1,
    random_state=42,
    eval_metric='logloss'
)

xgb_grid = GridSearchCV(
    xgb_base,
    xgb_params,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1,
    refit=True
)

start = time.time()
xgb_grid.fit(X_train, y_train)
xgb_time = time.time() - start

print(f"\nXGBoost Grid Search complete in {xgb_time:.1f}s")
print(f"Best parameters: {xgb_grid.best_params_}")
print(f"Best CV F1:      {xgb_grid.best_score_:.4f}")
print(f"Your current:    n_estimators=300, max_depth=8, lr=0.1")

# ── Random Forest Grid Search ──────────────────────────────
print("\n" + "=" * 60)
print("RANDOM FOREST HYPERPARAMETER TUNING")
print("=" * 60)
print("Testing parameter combinations (5-fold CV)...")
print("This may take 10-20 minutes...\n")

rf_params = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth':    [10, 15, 20, 30],
    'max_features': ['sqrt', 'log2', 0.5],
}

rf_base = RandomForestClassifier(
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)

rf_grid = GridSearchCV(
    rf_base,
    rf_params,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1,
    refit=True
)

start = time.time()
rf_grid.fit(X_train, y_train)
rf_time = time.time() - start

print(f"\nRF Grid Search complete in {rf_time:.1f}s")
print(f"Best parameters: {rf_grid.best_params_}")
print(f"Best CV F1:      {rf_grid.best_score_:.4f}")
print(f"Your current:    n_estimators=300, max_depth=20, max_features=sqrt")

# ── Results comparison ─────────────────────────────────────
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

xgb_results = pd.DataFrame(xgb_grid.cv_results_)[
    ['param_n_estimators','param_max_depth','param_learning_rate',
     'mean_test_score','std_test_score','rank_test_score']
].sort_values('rank_test_score').head(10)

rf_results = pd.DataFrame(rf_grid.cv_results_)[
    ['param_n_estimators','param_max_depth','param_max_features',
     'mean_test_score','std_test_score','rank_test_score']
].sort_values('rank_test_score').head(10)

print("\nTop 10 XGBoost configurations:")
print(xgb_results.to_string(index=False))

print("\nTop 10 Random Forest configurations:")
print(rf_results.to_string(index=False))

# ── Save results ───────────────────────────────────────────
output = {
    'xgboost': {
        'best_params': xgb_grid.best_params_,
        'best_cv_f1': round(xgb_grid.best_score_, 4),
        'search_time_seconds': round(xgb_time, 1),
        'your_current_params': {'n_estimators': 300, 'max_depth': 8, 'learning_rate': 0.1},
        'top10_results': xgb_results.to_dict(orient='records')
    },
    'random_forest': {
        'best_params': rf_grid.best_params_,
        'best_cv_f1': round(rf_grid.best_score_, 4),
        'search_time_seconds': round(rf_time, 1),
        'your_current_params': {'n_estimators': 300, 'max_depth': 20},
        'top10_results': rf_results.to_dict(orient='records')
    }
}

# Convert numpy types for JSON
def convert(o):
    if isinstance(o, (np.integer, np.floating)):
        return float(o)
    return str(o)

with open('reports/hyperparameter_results.json', 'w') as f:
    json.dump(output, f, indent=2, default=convert)

print(f"\nSaved: reports/hyperparameter_results.json")

print("\n" + "=" * 60)
print("THESIS STATEMENT")
print("=" * 60)
xgb_current_rank = None
for _, row in xgb_results.iterrows():
    if (int(row['param_n_estimators']) == 300 and
        int(row['param_max_depth']) == 8 and
        float(row['param_learning_rate']) == 0.1):
        xgb_current_rank = int(row['rank_test_score'])
        break

if xgb_current_rank == 1:
    print("XGBoost: Your current parameters ARE the optimal configuration.")
    print("Thesis: 'Grid search confirmed n_estimators=300, max_depth=8 as optimal.'")
elif xgb_current_rank and xgb_current_rank <= 3:
    print(f"XGBoost: Your parameters rank #{xgb_current_rank} — near-optimal.")
    print(f"Best params: {xgb_grid.best_params_}")
    print("Thesis: 'Parameters were within top-3 of grid search results.'")
else:
    print(f"XGBoost best params differ: {xgb_grid.best_params_}")
    print("Consider retraining with best params if F1 improvement > 0.001")

print("\n✅ Hyperparameter tuning complete.")
