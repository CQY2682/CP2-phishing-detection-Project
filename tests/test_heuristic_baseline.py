"""
Heuristic Engine Baseline Test
==============================

Purpose: Measure how many PhiUSIIL URLs Layer 1 (heuristics) catches
         BEFORE the ML ensemble (Layer 2) is needed.

Outputs:
    - Coverage stats (HIGH_RISK / SUSPICIOUS / CLEAN per class)
    - Per-rule trigger counts
    - False positive estimate (heuristics flagging legitimate URLs)
    - Per-rule MITRE technique frequencies

Run: python tests/test_heuristic_baseline.py
"""

import sys
import os
import time
from collections import Counter
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from heuristic_engine import heuristic_check, RULES


def main():
    print("=" * 70)
    print("HEURISTIC ENGINE — Baseline Test on Full PhiUSIIL")
    print("=" * 70)
    
    # Load original dataset (need raw URLs, not preprocessed features)
    print("\nLoading raw dataset...")
    df = pd.read_csv('data/PhiUSIIL_Phishing_URL_Dataset.csv', usecols=['URL', 'label'])
    # Apply same preprocessing as Week 3 (dedupe + flip labels)
    df = df.drop_duplicates(subset=['URL'], keep='first').reset_index(drop=True)
    df['label'] = 1 - df['label']  # 1=phishing now
    print(f"  {len(df):,} unique URLs (1=phishing, 0=legitimate)")
    
    # Run heuristic_check on all URLs
    print(f"\nRunning heuristic engine on {len(df):,} URLs...")
    start = time.time()
    
    verdicts = []
    rule_hits = Counter()
    technique_hits = Counter()
    
    for i, url in enumerate(df['URL']):
        verdict, rules, techniques = heuristic_check(url)
        verdicts.append(verdict)
        rule_hits.update(rules)
        technique_hits.update(techniques)
        if (i + 1) % 50000 == 0:
            print(f"  [{i+1:>6,}/{len(df):,}] {time.time()-start:.1f}s")
    
    df['verdict'] = verdicts
    total_time = time.time() - start
    print(f"  Done in {total_time:.1f}s ({len(df)/total_time:.0f} URLs/sec)")
    
    # ============================================================
    # 1. OVERALL COVERAGE
    # ============================================================
    print("\n" + "-" * 70)
    print("COVERAGE BREAKDOWN")
    print("-" * 70)
    
    verdict_dist = df['verdict'].value_counts()
    for v in ['HIGH_RISK', 'SUSPICIOUS', 'CLEAN']:
        count = verdict_dist.get(v, 0)
        pct = count / len(df) * 100
        print(f"  {v:12s}: {count:>7,} ({pct:>5.2f}%)")
    
    # ============================================================
    # 2. PER-CLASS BREAKDOWN
    # ============================================================
    print("\n" + "-" * 70)
    print("VERDICT BY ACTUAL CLASS (label)")
    print("-" * 70)
    
    crosstab = pd.crosstab(df['label'], df['verdict'], margins=True)
    # Reorder columns
    col_order = [c for c in ['HIGH_RISK', 'SUSPICIOUS', 'CLEAN', 'All']
                 if c in crosstab.columns]
    crosstab = crosstab[col_order]
    crosstab.index = ['Legitimate (0)', 'Phishing (1)', 'Total']
    print(crosstab)
    
    # ============================================================
    # 3. KEY METRICS
    # ============================================================
    print("\n" + "-" * 70)
    print("KEY METRICS")
    print("-" * 70)
    
    phishing_total = (df['label'] == 1).sum()
    legit_total = (df['label'] == 0).sum()
    
    phish_caught = ((df['label'] == 1) & (df['verdict'] == 'HIGH_RISK')).sum()
    phish_suspicious = ((df['label'] == 1) & (df['verdict'] == 'SUSPICIOUS')).sum()
    phish_missed = ((df['label'] == 1) & (df['verdict'] == 'CLEAN')).sum()
    
    legit_fp_high = ((df['label'] == 0) & (df['verdict'] == 'HIGH_RISK')).sum()
    legit_fp_suspicious = ((df['label'] == 0) & (df['verdict'] == 'SUSPICIOUS')).sum()
    legit_clean = ((df['label'] == 0) & (df['verdict'] == 'CLEAN')).sum()
    
    print(f"\n  Phishing detection (recall):")
    print(f"    HIGH_RISK   : {phish_caught:>6,} / {phishing_total:,} ({phish_caught/phishing_total*100:.2f}%)")
    print(f"    SUSPICIOUS  : {phish_suspicious:>6,} / {phishing_total:,} ({phish_suspicious/phishing_total*100:.2f}%)")
    print(f"    Missed (→ML): {phish_missed:>6,} / {phishing_total:,} ({phish_missed/phishing_total*100:.2f}%)")
    
    print(f"\n  Legitimate handling (false positive rate):")
    print(f"    HIGH_RISK   : {legit_fp_high:>6,} / {legit_total:,} ({legit_fp_high/legit_total*100:.4f}%) ← FP")
    print(f"    SUSPICIOUS  : {legit_fp_suspicious:>6,} / {legit_total:,} ({legit_fp_suspicious/legit_total*100:.4f}%)")
    print(f"    CLEAN (→ML) : {legit_clean:>6,} / {legit_total:,} ({legit_clean/legit_total*100:.2f}%)")
    
    print(f"\n  ML workload savings:")
    cleared = (df['verdict'] == 'HIGH_RISK').sum()
    pct_saved = cleared / len(df) * 100
    print(f"    URLs decided by Layer 1: {cleared:,} ({pct_saved:.2f}%)")
    print(f"    URLs passed to Layer 2:  {len(df) - cleared:,} ({100-pct_saved:.2f}%)")
    
    # ============================================================
    # 4. PER-RULE TRIGGER FREQUENCY
    # ============================================================
    print("\n" + "-" * 70)
    print("PER-RULE TRIGGER COUNTS")
    print("-" * 70)
    
    for rule_name in RULES:
        count = rule_hits.get(rule_name, 0)
        pct = count / len(df) * 100
        _, severity, _, _ = RULES[rule_name]
        print(f"  [{severity:11s}] {rule_name:25s}: {count:>7,} hits ({pct:.3f}%)")
    
    # ============================================================
    # 5. MITRE TECHNIQUE FREQUENCY
    # ============================================================
    print("\n" + "-" * 70)
    print("MITRE ATT&CK TECHNIQUE FREQUENCY")
    print("-" * 70)
    
    for technique, count in technique_hits.most_common():
        pct = count / len(df) * 100
        print(f"  {technique:12s}: {count:>7,} hits ({pct:.3f}%)")
    
    print("\n" + "=" * 70)
    print("BASELINE TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()