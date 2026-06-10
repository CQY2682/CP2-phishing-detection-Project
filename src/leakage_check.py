"""
CP2 Data Leakage Check (corrected — distinct counting)
Author: Cheah Qi Yang (22095483)
Module: src/leakage_check.py
Run: python src/leakage_check.py
"""

import pandas as pd
import json

def main():
    print("=" * 60)
    print("CP2 DATA LEAKAGE CHECK")
    print("=" * 60)

    X_train = pd.read_csv('data/X_train.csv')
    X_test  = pd.read_csv('data/X_test.csv')
    train   = pd.read_csv('data/y_train.csv')
    test    = pd.read_csv('data/y_test.csv')

    print(f"\n  Train rows: {len(X_train)}")
    print(f"  Test rows:  {len(X_test)}")

    # ── Distinct feature vectors ────────────────────────────
    train_vectors = set(map(tuple, X_train.values))
    test_vectors  = set(map(tuple, X_test.values))
    print(f"\n  Distinct feature vectors in train: {len(train_vectors)}")
    print(f"  Distinct feature vectors in test:  {len(test_vectors)}")

    # ── Real overlap: test ROWS whose vector also in train ──
    print("\n[1] Feature-space overlap (honest count)...")
    test_tuples = list(map(tuple, X_test.values))
    leaked = sum(1 for t in test_tuples if t in train_vectors)
    pct = 100 * leaked / len(X_test)
    print(f"  Test rows whose feature vector also appears in train: {leaked} ({pct:.1f}%)")

    # ── Internal duplicates ─────────────────────────────────
    print("\n[2] Internal duplicate feature rows...")
    print(f"  Train: {X_train.duplicated().sum()}  ({100*X_train.duplicated().mean():.1f}%)")
    print(f"  Test:  {X_test.duplicated().sum()}  ({100*X_test.duplicated().mean():.1f}%)")

    # ── Stratification ──────────────────────────────────────
    print("\n[3] Stratification...")
    print(f"  Train phishing ratio: {train['label'].mean():.4f}")
    print(f"  Test phishing ratio:  {test['label'].mean():.4f}")

    print("\n" + "=" * 60)
    print("LEAKAGE CHECK COMPLETE")
    print("=" * 60)

    results = {
        'train_rows': int(len(X_train)),
        'test_rows': int(len(X_test)),
        'distinct_train_vectors': len(train_vectors),
        'distinct_test_vectors': len(test_vectors),
        'leaked_test_rows': leaked,
        'leaked_test_pct': round(pct, 1),
        'train_duplicate_rows': int(X_train.duplicated().sum()),
        'train_duplicate_pct': round(100*X_train.duplicated().mean(), 1),
        'test_duplicate_rows': int(X_test.duplicated().sum()),
        'test_duplicate_pct': round(100*X_test.duplicated().mean(), 1),
        'train_phishing_ratio': round(float(train['label'].mean()), 4),
        'test_phishing_ratio': round(float(test['label'].mean()), 4),
    }
    with open('reports/leakage_check.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n  Saved: reports/leakage_check.json")

if __name__ == "__main__":
    main()