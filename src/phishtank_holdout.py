"""
CP2 Milestone E: OpenPhish Blind Hold-out Test
Author: Cheah Qi Yang (22095483)

Purpose:
    Evaluate hybrid detection system on unseen real-world phishing URLs.
    These URLs were NEVER used during training — true generalization test.

Run: python src/phishtank_holdout.py
"""

import os, sys, time
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hybrid_scorer import hybrid_predict

INPUT  = 'data/openphish.txt'
OUTPUT = 'reports/openphish_holdout_results.csv'


def main():
    print("=" * 65)
    print("MILESTONE E: OpenPhish Blind Hold-out Evaluation")
    print("=" * 65)

    # Load URLs
    with open(INPUT, 'r', encoding='utf-8', errors='ignore') as f:
        urls = [u.strip() for u in f if u.strip() and u.startswith('http')]
    print(f"\nLoaded {len(urls):,} URLs from OpenPhish feed")

    # Run pipeline
    print(f"Running hybrid detection on all URLs...")
    start = time.time()

    results = []
    verdicts = {'CRITICAL':0,'HIGH':0,'MEDIUM':0,'LOW':0,'SAFE':0}

    for i, url in enumerate(urls):
        try:
            r = hybrid_predict(url)
            verdicts[r['verdict']] += 1
            results.append({
                'url':             url[:150],
                'verdict':         r['verdict'],
                'score':           r['score'],
                'cvss':            r['cvss'],
                'layer1':          r['layer1_verdict'],
                'layer2_score':    r['layer2_score'],
                'escalated':       r['escalated'],
                'mitre':           ','.join(r['mitre_techniques']),
                'rules':           ','.join(r['layer1_rules']),
            })
        except Exception as e:
            results.append({
                'url': url[:150], 'verdict': 'ERROR',
                'score': None, 'cvss': None,
                'layer1': 'ERROR', 'layer2_score': None,
                'escalated': False, 'mitre': '', 'rules': str(e)[:50],
            })

        if (i+1) % 100 == 0:
            elapsed = time.time() - start
            print(f"  [{i+1:>4}/{len(urls)}] {elapsed:.1f}s")

    elapsed = time.time() - start
    df = pd.DataFrame(results)
    os.makedirs('reports', exist_ok=True)
    df.to_csv(OUTPUT, index=False)

    # ── Results ──────────────────────────────────────────
    total     = len(df)
    errors    = (df['verdict'] == 'ERROR').sum()
    valid     = total - errors
    detected  = df[df['verdict'].isin(['CRITICAL','HIGH','MEDIUM'])].shape[0]
    missed    = df[df['verdict'].isin(['LOW','SAFE'])].shape[0]

    print("\n" + "=" * 65)
    print("RESULTS")
    print("=" * 65)
    print(f"\n  Total URLs:      {total:,}")
    print(f"  Valid processed: {valid:,}")
    print(f"  Errors:          {errors:,}")

    print(f"\n  Verdict distribution:")
    for v, count in verdicts.items():
        pct = count/total*100 if total > 0 else 0
        bar = '█' * int(pct/2)
        print(f"    {v:8s}: {count:>5,} ({pct:>5.1f}%) {bar}")

    detection_rate = detected / valid * 100 if valid > 0 else 0
    print(f"\n  Detection rate:  {detection_rate:.2f}% ({detected:,}/{valid:,})")
    print(f"  Missed:          {100-detection_rate:.2f}% ({missed:,}/{valid:,})")

    print(f"\n  Layer 1 breakdown:")
    l1 = df['layer1'].value_counts()
    for k, v in l1.items():
        print(f"    {k:12s}: {v:,} ({v/total*100:.1f}%)")

    print(f"\n  Speed: {total/elapsed:.0f} URLs/sec")
    print(f"\n  Saved: {OUTPUT}")
    print("\n[DONE] Milestone E complete.")


if __name__ == "__main__":
    main()