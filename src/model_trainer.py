"""
CP2 Model Trainer
=================

Author: Cheah Qi Yang (22095483)
Module: src/model_trainer.py

Purpose:
    Train XGBoost + Random Forest ensemble on PhiUSIIL features.
    Compare F1 scores to justify ensemble weights for hybrid scorer.

Inputs:
    data/X_train.csv, data/X_test.csv, data/y_train.csv, data/y_test.csv

Outputs:
    models/xgboost_model.joblib
    models/rf_model.joblib
    reports/training_metrics.json
    reports/feature_importances.csv

Usage:
    python src/model_trainer.py
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score
)
import xgboost as xgb


# ============================================================
# CONSTANTS
# ============================================================
RANDOM_STATE = 42

# XGBoost hyperparameters
XGB_PARAMS = {
    'n_estimators': 300,
    'max_depth': 8,
    'learning_rate': 0.1,
    'n_jobs': -1,           # use all CPU cores
    'random_state': RANDOM_STATE,
    'eval_metric': 'logloss',
    'tree_method': 'hist',  # faster training
}

# Random Forest hyperparameters
RF_PARAMS = {
    'n_estimators': 300,
    'max_depth': 20,
    'n_jobs': -1,
    'random_state': RANDOM_STATE,
    'class_weight': 'balanced',  # handles mild 57/43 imbalance
}


# ============================================================
# HELPERS
# ============================================================

def load_splits(augmented=False):
    """Load preprocessed train/test data."""
    print("\n[1/5] Loading preprocessed splits...")
    if augmented and os.path.exists('data/X_train_aug.csv'):
        X_train = pd.read_csv('data/X_train_aug.csv')
        y_train = pd.read_csv('data/y_train_aug.csv')['label']
        print(f"  Using AUGMENTED training data")
    else:
        X_train = pd.read_csv('data/X_train.csv')
        y_train = pd.read_csv('data/y_train.csv')['label']
    X_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv')['label']
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    return X_train, X_test, y_train, y_test


def evaluate_model(name, model, X_test, y_test):
    """Compute metrics and print classification report."""
    print(f"\n  Evaluating {name}...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
    }
    
    print(f"  {name} Test Metrics:")
    for k, v in metrics.items():
        print(f"    {k:12s}: {v:.4f}")
    
    print(f"\n  {name} Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                  Predicted")
    print(f"                  Legit  Phish")
    print(f"    Actual Legit  {cm[0][0]:6d}  {cm[0][1]:5d}")
    print(f"           Phish  {cm[1][0]:6d}  {cm[1][1]:5d}")
    
    print(f"\n  {name} Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))
    
    return metrics


def compute_ensemble_weights(xgb_f1, rf_f1):
    """
    Compute ensemble weights from F1 scores.
    Better F1 → higher weight.
    Justifies CP1 promise of weight justification.
    """
    total = xgb_f1 + rf_f1
    w_xgb = xgb_f1 / total
    w_rf = rf_f1 / total
    return round(w_xgb, 4), round(w_rf, 4)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("CP2 MODEL TRAINER")
    print("=" * 70)
    total_start = time.time()
    
    # Ensure models/ and reports/ exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    # Load data
    augmented = '--augmented' in sys.argv
    X_train, X_test, y_train, y_test = load_splits(augmented=augmented)
    
    # ============================================================
    # XGBOOST
    # ============================================================
    print("\n[2/5] Training XGBoost...")
    print(f"  Params: {XGB_PARAMS}")
    
    start = time.time()
    xgb_model = xgb.XGBClassifier(**XGB_PARAMS)
    xgb_model.fit(X_train, y_train)
    xgb_train_time = time.time() - start
    print(f"  Trained in {xgb_train_time:.1f}s")
    
    xgb_metrics = evaluate_model("XGBoost", xgb_model, X_test, y_test)
    
    # Save model
    joblib.dump(xgb_model, 'models/xgboost_model.joblib')
    print("  Saved: models/xgboost_model.joblib")
    
    # ============================================================
    # RANDOM FOREST
    # ============================================================
    print("\n[3/5] Training Random Forest...")
    print(f"  Params: {RF_PARAMS}")
    
    start = time.time()
    rf_model = RandomForestClassifier(**RF_PARAMS)
    rf_model.fit(X_train, y_train)
    rf_train_time = time.time() - start
    print(f"  Trained in {rf_train_time:.1f}s")
    
    rf_metrics = evaluate_model("Random Forest", rf_model, X_test, y_test)
    
    joblib.dump(rf_model, 'models/rf_model.joblib')
    print("  Saved: models/rf_model.joblib")
    
    # ============================================================
    # ENSEMBLE WEIGHTS (justify by F1)
    # ============================================================
    print("\n[4/5] Computing ensemble weights from F1 scores...")
    w_xgb, w_rf = compute_ensemble_weights(xgb_metrics['f1'], rf_metrics['f1'])
    print(f"  XGBoost F1:       {xgb_metrics['f1']:.4f}")
    print(f"  Random Forest F1: {rf_metrics['f1']:.4f}")
    print(f"  → XGBoost weight: {w_xgb:.4f}")
    print(f"  → Random Forest weight: {w_rf:.4f}")
    print(f"\n  Hybrid score = ({w_xgb:.2f} × XGB_proba) + ({w_rf:.2f} × RF_proba)")
    
    # ============================================================
    # FEATURE IMPORTANCES
    # ============================================================
    print("\n[5/5] Feature importances (XGBoost)...")
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'xgb_importance': xgb_model.feature_importances_,
        'rf_importance': rf_model.feature_importances_,
    }).sort_values('xgb_importance', ascending=False)
    
    print("\n  Top 10 features by XGBoost importance:")
    for _, row in importance_df.head(10).iterrows():
        bar = '█' * int(row['xgb_importance'] * 100)
        print(f"    {row['feature']:25s} {row['xgb_importance']:.4f}  {bar}")
    
    importance_df.to_csv('reports/feature_importances.csv', index=False)
    print("\n  Saved: reports/feature_importances.csv")
    
    # ============================================================
    # SAVE TRAINING METRICS JSON
    # ============================================================
    metrics_output = {
        'training_date': time.strftime('%Y-%m-%d'),
        'xgboost': {
            **xgb_metrics,
            'train_time_seconds': round(xgb_train_time, 2),
            'hyperparameters': XGB_PARAMS,
        },
        'random_forest': {
            **rf_metrics,
            'train_time_seconds': round(rf_train_time, 2),
            'hyperparameters': RF_PARAMS,
        },
        'ensemble_weights': {
            'xgboost_weight': w_xgb,
            'rf_weight': w_rf,
            'justification': 'Weights proportional to F1 scores (CP1 promise)',
        },
        'data_splits': {
            'train_size': len(X_train),
            'test_size': len(X_test),
            'features': list(X_train.columns),
        }
    }
    
    # Convert numpy types for JSON serialization
    def convert(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o
    
    metrics_output_serializable = json.loads(json.dumps(metrics_output, default=convert))
    
    with open('reports/training_metrics.json', 'w') as f:
        json.dump(metrics_output_serializable, f, indent=2)
    print("  Saved: reports/training_metrics.json")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    total_time = time.time() - total_start
    print("\n" + "=" * 70)
    print("MODEL TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Total time: {total_time:.1f}s")
    print(f"  XGBoost F1: {xgb_metrics['f1']:.4f}")
    print(f"  Random Forest F1: {rf_metrics['f1']:.4f}")
    print(f"  Ensemble weights: XGB={w_xgb}, RF={w_rf}")
    print(f"\n  Models saved to: models/")
    print(f"  Metrics saved to: reports/training_metrics.json")
    print(f"  Importances saved to: reports/feature_importances.csv")
    print(f"\n  Ready for: Milestone B (hybrid scorer)")


if __name__ == "__main__":
    main()