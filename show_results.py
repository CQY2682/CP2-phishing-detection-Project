"""
show_results.py  -  Print every headline result in one place.

READ-ONLY. Does not train, evaluate, or change anything.
It only reads the existing files in reports/ and prints a summary,
so you can sanity-check that all numbers agree and show them fast in a viva.

Run from the project root:
    python show_results.py
"""

import json
import os
import csv

R = "reports"


def load_json(name):
    with open(os.path.join(R, name), encoding="utf-8") as f:
        return json.load(f)


def line(c="-", n=64):
    print(c * n)


def head(title):
    print()
    line("=")
    print("  " + title)
    line("=")


def main():
    print()
    line("=")
    print("  PhishLens  -  Results Summary  (read-only)")
    line("=")

    # 1. Metrics -------------------------------------------------------
    ev = load_json("full_evaluation.json")
    head("1. TEST-SET METRICS  (full_evaluation.json)")
    print(f"  {'Model':22s}{'Acc':>8}{'Prec':>9}{'Rec':>9}{'F1':>9}{'ROC-AUC':>10}{'MCC':>9}")
    line("-")
    for m in ev["metrics"]:
        nm = m.get("model", "?")
        print(f"  {nm:22s}{m.get('accuracy',0):>8.4f}{m.get('precision',0):>9.4f}"
              f"{m.get('recall',0):>9.4f}{m.get('f1',0):>9.4f}"
              f"{m.get('roc_auc',0):>10.4f}{m.get('mcc',0):>9.4f}")

    # 2. Confusion -----------------------------------------------------
    head("2. CONFUSION MATRICES  (rows: [[TN, FP], [FN, TP]])")
    for name, cm in ev["confusion_matrices"].items():
        (tn, fp), (fn, tp) = cm
        print(f"  {name:14s} TN={tn:>6}  FP={fp:>4}  FN={fn:>4}  TP={tp:>6}")

    # 3. McNemar -------------------------------------------------------
    head("3. McNEMAR TEST  (p >= 0.05 => no significant difference)")
    for t in ev["mcnemar_tests"]:
        sig = "significant" if t.get("significant") else "not significant"
        print(f"  {t['comparison']:28s} p={t['p_value']:.4f}  ({sig})")

    # 4. Hyperparameters ----------------------------------------------
    hp = load_json("hyperparameter_results.json")
    head("4. HYPERPARAMETER SEARCH  (48 combos, 5-fold CV)")
    for m in ["xgboost", "random_forest"]:
        b = hp[m]
        print(f"  {m:14s} best={b.get('best_params')}  CV-F1={b.get('best_cv_f1')}")

    # 5. Overfitting ---------------------------------------------------
    tvt = load_json("train_val_test_results.json")
    head("5. OVERFITTING / SPLITS  (train_val_test_results.json)")
    s = tvt["split_sizes"]
    print(f"  Train {s['train']} ({s['train_pct']}%) | "
          f"Val {s['validation']} ({s['val_pct']}%) | "
          f"Test {s['test']} ({s['test_pct']}%)")
    o = tvt["overfitting_analysis"]
    print(f"  XGB train-val gap={o['xgboost_train_val_gap']}  "
          f"RF gap={o['rf_train_val_gap']}  => {o['conclusion']}")

    # 6. Collision -----------------------------------------------------
    col = load_json("collision_analysis.json")
    head("6. COLLISION / DATA INTEGRITY")
    cf = col["collision_free_metrics"]
    if "hybrid" in cf:
        cf = cf["hybrid"]
    cf_f1 = cf.get("f1", cf.get("f1_score", "?"))
    print(f"  Feature overlap: {col['feature_overlap_pct']}%  |  "
          f"collision-free rows: {col['collision_free_rows']} ({col['collision_free_pct']}%)")
    print(f"  Collision-free F1: {cf_f1}   (full-set F1: 0.9976)")

    lk = load_json("leakage_check.json")
    print(f"  Leaked test rows: {lk['leaked_test_pct']}%  "
          f"(train {lk['train_rows']}, test {lk['test_rows']})")

    # 7. Weights -------------------------------------------------------
    tm = load_json("training_metrics.json")
    head("7. ENSEMBLE WEIGHTS")
    w = tm["ensemble_weights"]
    print(f"  XGBoost weight={w['xgboost_weight']}  RF weight={w['rf_weight']}")
    print(f"  Features: {len(tm['data_splits']['features'])}  "
          f"| augmented train size: {tm['data_splits']['train_size']}")

    # 8. Adversarial & OpenPhish (CSV) --------------------------------
    head("8. ADVERSARIAL & OPENPHISH  (CSV)")
    try:
        with open(os.path.join(R, "adversarial_results.csv"), encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        col_ev = next((c for c in rows[0] if "evad" in c.lower()), None)
        evasions = sum(1 for r in rows if col_ev and str(r[col_ev]).lower() in ("true", "1")) if col_ev else "?"
        print(f"  Adversarial: {len(rows)} tests, evasions={evasions}")
    except Exception as e:
        print(f"  Adversarial: (skip: {e})")
    try:
        with open(os.path.join(R, "openphish_holdout_results.csv"), encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        vk = "verdict" if rows and "verdict" in rows[0] else None
        det = sum(1 for r in rows if vk and r[vk] in ("CRITICAL", "HIGH", "MEDIUM"))
        print(f"  OpenPhish: {len(rows)} URLs, detected={det}")
    except Exception as e:
        print(f"  OpenPhish: (skip: {e})")

    print()
    line("=")
    print("  End of summary. Read-only; nothing was changed.")
    line("=")
    print()


if __name__ == "__main__":
    main()