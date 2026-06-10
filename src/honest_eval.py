"""
CP2 Honest Evaluation — collision-free test subset
Author: Cheah Qi Yang (22095483)
Module: src/honest_eval.py
Run: python src/honest_eval.py
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score

W_XGB, W_RF = 0.5001, 0.4999

def main():
    X_train = pd.read_csv('data/X_train.csv')
    X_test  = pd.read_csv('data/X_test.csv')
    y_test  = pd.read_csv('data/y_test.csv')['label']
    xgb = joblib.load('models/xgboost_model.joblib')
    rf  = joblib.load('models/rf_model.joblib')

    train_vectors = set(map(tuple, X_train.values))
    mask = np.array([tuple(r) not in train_vectors for r in X_test.values])

    print(f"Total test rows:        {len(X_test)}")
    print(f"Collision-free rows:    {mask.sum()} ({100*mask.mean():.1f}%)")

    Xc, yc = X_test[mask], y_test[mask]

    xgb_p = xgb.predict_proba(Xc)[:,1]
    rf_p  = rf.predict_proba(Xc)[:,1]
    hyb_p = W_XGB*xgb_p + W_RF*rf_p

    for name, proba in [("XGBoost", xgb_p), ("Random Forest", rf_p), ("Hybrid", hyb_p)]:
        pred = (proba >= 0.5).astype(int)
        print(f"\n{name} — collision-free subset:")
        print(f"  Accuracy:  {accuracy_score(yc, pred):.4f}")
        print(f"  Precision: {precision_score(yc, pred):.4f}")
        print(f"  Recall:    {recall_score(yc, pred):.4f}")
        print(f"  F1:        {f1_score(yc, pred):.4f}")
        print(f"  ROC-AUC:   {roc_auc_score(yc, proba):.4f}")

    import json
    results = {
        'total_test_rows': int(len(X_test)),
        'collision_free_rows': int(mask.sum()),
        'collision_free_pct': round(100*mask.mean(), 1),
        'feature_overlap_pct': round(100*(1 - mask.mean()), 1),
        'collision_free_metrics': {
            'hybrid': {
                'accuracy':  round(accuracy_score(yc, (hyb_p>=0.5).astype(int)), 4),
                'precision': round(float(precision_score(yc, (hyb_p>=0.5).astype(int))), 4),
                'recall':    round(float(recall_score(yc, (hyb_p>=0.5).astype(int))), 4),
                'f1':        round(float(f1_score(yc, (hyb_p>=0.5).astype(int))), 4),
                'roc_auc':   round(roc_auc_score(yc, hyb_p), 4),
            }
        }
    }
    with open('reports/collision_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n  Saved: reports/collision_analysis.json")

if __name__ == "__main__":
    main()