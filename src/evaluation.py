"""
CP2 Full Evaluation (Post-Supervisor Feedback)
================================================
Author: Cheah Qi Yang (22095483)
Purpose:
    Generate complete evaluation metrics beyond F1:
    - Precision, Recall, F1, Accuracy, ROC-AUC, MCC
    - Confusion matrix figures
    - ROC curve figures
    - McNemar's test (hybrid vs ML-only)

Run: python src/evaluation.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix, roc_curve, classification_report,
    ConfusionMatrixDisplay
)
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Constants ──────────────────────────────────────────────
W_XGB = 0.5001
W_RF  = 0.4999
FIGURES_DIR = 'reports/figures'
os.makedirs(FIGURES_DIR, exist_ok=True)


# ── Load data and models ───────────────────────────────────
def load_all():
    print("Loading data and models...")
    X_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv')['label']
    xgb    = joblib.load('models/xgboost_model.joblib')
    rf     = joblib.load('models/rf_model.joblib')
    print(f"  Test set: {X_test.shape}")
    return X_test, y_test, xgb, rf


# ── Compute all metrics ────────────────────────────────────
def compute_metrics(name, y_true, y_pred, y_proba):
    return {
        'model':     name,
        'accuracy':  round(accuracy_score(y_true, y_pred), 4),
        'precision': round(float(precision_score(y_true, y_pred)), 4),
        'recall':    round(float(recall_score(y_true, y_pred)), 4),
        'f1':        round(float(f1_score(y_true, y_pred)), 4),
        'roc_auc':   round(roc_auc_score(y_true, y_proba), 4),
        'mcc':       round(matthews_corrcoef(y_true, y_pred), 4),
    }


# ── Confusion matrix figure ────────────────────────────────
def plot_confusion_matrix(name, y_true, y_pred, filename):
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['Legitimate', 'Phishing']
    )
    disp.plot(ax=ax, colorbar=False, cmap='Blues', values_format='d')
    ax.set_title(f'Confusion Matrix — {name}', fontsize=13, pad=12)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ── ROC curve figure ───────────────────────────────────────
def plot_roc_curves(y_true, probas_dict):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0e1117')

    colors = {'XGBoost': '#1F4E79', 'Random Forest': '#2E75B6', 'Hybrid Ensemble': '#C00000'}
    styles = {'XGBoost': '-', 'Random Forest': '--', 'Hybrid Ensemble': '-.'}
    widths = {'XGBoost': 3, 'Random Forest': 3, 'Hybrid Ensemble': 2.5}

    ax.set_facecolor('#0e1117')
    for spine in ax.spines.values():
        spine.set_color('#444')
    ax.tick_params(colors='#aaa', labelsize=10)
    ax.set_xlabel('False Positive Rate', fontsize=11, color='#ccc', labelpad=8)
    ax.set_ylabel('True Positive Rate', fontsize=11, color='#ccc', labelpad=8)
    ax.set_title('ROC curves — zoomed (FPR 0–1%)', fontsize=13, color='white', pad=16)

    auc_parts = []
    for name, proba in probas_dict.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc = roc_auc_score(y_true, proba)
        auc_parts.append(f'{name} {auc:.4f}')
        ax.plot(fpr, tpr,
                linestyle=styles[name],
                linewidth=widths[name],
                color=colors[name],
                label=f'{name}  (AUC = {auc:.4f})')

    subtitle = 'Full-range view omitted — all AUC ≈ 0.999 (near-ceiling). ' + ' · '.join(auc_parts)
    ax.text(0.5, 1.01, subtitle, transform=ax.transAxes,
            fontsize=9, color='#888', ha='center', va='bottom')

    ax.set_xlim((0, 0.01))
    ax.set_ylim((0.990, 1.001))
    ax.grid(True, alpha=0.2, color='#555', linewidth=0.8)

    ax.legend(loc='lower right', fontsize=10,
              framealpha=0.3, edgecolor='#555',
              fancybox=False, labelcolor='white')

    fig.suptitle('XGBoost vs Random Forest vs Hybrid Ensemble',
                 fontsize=14, color='white', y=0.98)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'roc_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0e1117')
    plt.close()
    print(f"  Saved: {path}")


# ── Metric comparison bar chart ────────────────────────────
def plot_metric_comparison(metrics_list):
    metric_names = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    model_names = [m['model'] for m in metrics_list]

    bar_colors = {
        'Random Baseline':     '#777777',
        'Logistic Regression': '#E8A33D',
        'XGBoost':             '#1F4E79',
        'Random Forest':       '#2E75B6',
        'Hybrid Ensemble':     '#C00000',
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    for spine in ax.spines.values():
        spine.set_color('#444')
    ax.tick_params(colors='#aaa', labelsize=10)
    ax.set_ylabel('Score', fontsize=11, color='#ccc', labelpad=8)
    ax.set_title('Model Metric Comparison', fontsize=13, color='white', pad=16)

    n_groups = len(metric_names)
    n_models = len(model_names)
    bar_width = 0.8 / n_models
    x = np.arange(n_groups)

    for i, m in enumerate(metrics_list):
        values = [m[metric] for metric in metric_names]
        offset = (i - (n_models - 1) / 2) * bar_width
        ax.bar(x + offset, values, bar_width,
               label=m['model'], color=bar_colors.get(m['model'], '#888888'))

    ax.set_xticks(x)
    ax.set_xticklabels([n.upper() for n in metric_names], color='#ccc')
    ax.set_ylim((0, 1.05))
    ax.grid(True, axis='y', alpha=0.2, color='#555', linewidth=0.8)

    ax.legend(loc='lower right', fontsize=10,
              framealpha=0.3, edgecolor='#555',
              fancybox=False, labelcolor='white')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'metric_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0e1117')
    plt.close()
    print(f"  Saved: {path}")


# ── McNemar's test ─────────────────────────────────────────
def run_mcnemar(y_true, pred_a, pred_b, name_a, name_b):
    """
    McNemar's test: are two classifiers significantly different?
    Tests on cases where they DISAGREE.
    p < 0.05 = statistically significant difference.
    """
    # Build contingency table
    # b = A correct, B wrong
    # c = A wrong, B correct
    a_correct = (pred_a == y_true)
    b_correct = (pred_b == y_true)

    b_count = np.sum(a_correct & ~b_correct)   # A right, B wrong
    c_count = np.sum(~a_correct & b_correct)   # A wrong, B right

    table = [[np.sum(a_correct & b_correct),  b_count],
             [c_count, np.sum(~a_correct & ~b_correct)]]

    result = mcnemar(table, exact=False, correction=True)  # type: ignore

    # Safely coerce statsmodels result attributes to floats to satisfy type checkers
    stat = float(getattr(result, 'statistic', 0))
    pval = float(getattr(result, 'pvalue', 1))

    print(f"\n  McNemar's Test: {name_a} vs {name_b}")
    print(f"    Cases where {name_a} correct, {name_b} wrong: {b_count}")
    print(f"    Cases where {name_a} wrong, {name_b} correct: {c_count}")
    print(f"    Chi-squared statistic: {stat:.4f}")
    print(f"    p-value: {pval:.6f}")
    if pval < 0.05:
        print("    Result: SIGNIFICANT difference (p < 0.05)")
        if b_count > c_count:
            print(f"    {name_a} is significantly better than {name_b}")
        else:
            print(f"    {name_b} is significantly better than {name_a}")
    else:
        print("    Result: No significant difference (p >= 0.05)")

    return {
        'comparison': f'{name_a} vs {name_b}',
        'b_count': int(b_count),
        'c_count': int(c_count),
        'chi_squared': round(stat, 4),
        'p_value': round(pval, 6),
        'significant': bool(pval < 0.05)
    }


# ── Main ───────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("CP2 FULL EVALUATION")
    print("=" * 65)

    X_test, y_test, xgb_model, rf_model = load_all()

    # ── Get predictions ──────────────────────────────────
    print("\nGenerating predictions...")

    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    xgb_pred  = (xgb_proba >= 0.5).astype(int)

    rf_proba  = rf_model.predict_proba(X_test)[:, 1]
    rf_pred   = (rf_proba >= 0.5).astype(int)

    hybrid_proba = W_XGB * xgb_proba + W_RF * rf_proba
    hybrid_pred  = (hybrid_proba >= 0.5).astype(int)

    y_train = pd.read_csv('data/y_train.csv')['label']
    dummy = DummyClassifier(strategy='stratified', random_state=42)
    dummy.fit(np.zeros((len(y_train), 1)), y_train)
    dummy_pred  = dummy.predict(np.zeros((len(y_test), 1)))
    dummy_proba = np.asarray(dummy.predict_proba(np.zeros((len(y_test), 1))))[:, 1]

    # ── Logistic Regression baseline (real simple model) ──
    X_train_full = pd.read_csv('data/X_train.csv')
    scaler = StandardScaler().fit(X_train_full)
    logreg = LogisticRegression(max_iter=1000, random_state=42)
    logreg.fit(scaler.transform(X_train_full), y_train)
    logreg_proba = np.asarray(logreg.predict_proba(scaler.transform(X_test)))[:, 1]
    logreg_pred  = (logreg_proba >= 0.5).astype(int)

    print("  Done.")

    # ── Metrics table ────────────────────────────────────
    print("\n" + "=" * 65)
    print("METRICS COMPARISON")
    print("=" * 65)

    metrics = [
        compute_metrics("Random Baseline",     y_test, dummy_pred,  dummy_proba),
        compute_metrics("Logistic Regression", y_test, logreg_pred, logreg_proba),
        compute_metrics("XGBoost",             y_test, xgb_pred,    xgb_proba),
        compute_metrics("Random Forest",       y_test, rf_pred,     rf_proba),
        compute_metrics("Hybrid Ensemble",     y_test, hybrid_pred, hybrid_proba),
    ]

    df_metrics = pd.DataFrame(metrics).set_index('model')
    print(f"\n{df_metrics.to_string()}")

    # ── Classification reports ───────────────────────────
    print("\n" + "=" * 65)
    print("CLASSIFICATION REPORTS")
    print("=" * 65)
    for name, pred in [("XGBoost", xgb_pred), ("Random Forest", rf_pred), ("Hybrid", hybrid_pred)]:
        print(f"\n--- {name} ---")
        print(classification_report(y_test, pred, target_names=['Legitimate','Phishing']))

    # ── Confusion matrices ───────────────────────────────
    print("\n" + "=" * 65)
    print("CONFUSION MATRICES")
    print("=" * 65)
    plot_confusion_matrix("XGBoost",         y_test, xgb_pred,    "confusion_xgboost.png")
    plot_confusion_matrix("Random Forest",   y_test, rf_pred,     "confusion_rf.png")
    plot_confusion_matrix("Hybrid Ensemble", y_test, hybrid_pred, "confusion_hybrid.png")

    # ── ROC curves ───────────────────────────────────────
    print("\n" + "=" * 65)
    print("ROC CURVES")
    print("=" * 65)
    plot_roc_curves(y_test, {
        'XGBoost':         xgb_proba,
        'Random Forest':   rf_proba,
        'Hybrid Ensemble': hybrid_proba,
    })

    plot_metric_comparison(metrics)

    # ── McNemar's tests ──────────────────────────────────
    print("\n" + "=" * 65)
    print("McNEMAR'S STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 65)
    print("  (Tests whether performance differences are statistically significant)")

    mcnemar_results = []
    mcnemar_results.append(run_mcnemar(y_test, hybrid_pred, xgb_pred,  "Hybrid", "XGBoost"))
    mcnemar_results.append(run_mcnemar(y_test, hybrid_pred, rf_pred,   "Hybrid", "Random Forest"))
    mcnemar_results.append(run_mcnemar(y_test, xgb_pred,   rf_pred,   "XGBoost", "Random Forest"))

    # ── Save full results ────────────────────────────────
    print("\n" + "=" * 65)
    print("SAVING RESULTS")
    print("=" * 65)

    output = {
        'metrics': df_metrics.reset_index().to_dict(orient='records'),
        'mcnemar_tests': mcnemar_results,
        'confusion_matrices': {
            'xgboost':    confusion_matrix(y_test, xgb_pred).tolist(),
            'rf':         confusion_matrix(y_test, rf_pred).tolist(),
            'hybrid':     confusion_matrix(y_test, hybrid_pred).tolist(),
        }
    }

    path = 'reports/full_evaluation.json'
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {path}")
    print(f"  Figures: {FIGURES_DIR}/")

    print("\n" + "=" * 65)
    print("EVALUATION COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()