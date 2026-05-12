"""
Integration Test: Run extract_features() on full PhiUSIIL dataset.

Purpose: Verify all 24 features compute without crashes on real data,
         measure performance, and confirm statistics match Week 1 findings.

Run: python tests/test_integration_full.py
"""

import sys
import time
import os
import pandas as pd
import numpy as np

# Add src to path so we can import feature_extractor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from feature_extractor import extract_features, FEATURE_NAMES


def main():
    print("=" * 70)
    print("INTEGRATION TEST — Full PhiUSIIL Dataset")
    print("=" * 70)
    
    # Load
    print("\nLoading dataset...")
    start = time.time()
    df = pd.read_csv('data/PhiUSIIL_Phishing_URL_Dataset.csv')
    print(f"  Loaded {len(df):,} rows in {time.time()-start:.2f}s")
    
    # Extract features on all URLs
    print(f"\nExtracting {len(FEATURE_NAMES)} features for {len(df):,} URLs...")
    print("This will take 3-8 minutes on a 32GB machine.")
    
    start = time.time()
    features_list = []
    errors = []
    
    for i, url in enumerate(df['URL']):
        try:
            features = extract_features(url)
            features_list.append(features)
        except Exception as e:
            errors.append((i, url[:80], str(e)))
            # Append zero vector so DataFrame stays aligned
            features_list.append({f: 0 for f in FEATURE_NAMES})
        
        # Progress every 25k
        if (i + 1) % 25000 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (len(df) - i - 1) / rate
            print(f"  [{i+1:>6,}/{len(df):,}] {elapsed:.1f}s elapsed, "
                  f"{rate:.0f} URLs/sec, ETA {eta:.0f}s")
    
    total_time = time.time() - start
    print(f"\nDone in {total_time:.1f}s ({len(df)/total_time:.0f} URLs/sec)")
    print(f"Errors caught: {len(errors)}")
    
    # Build feature DataFrame
    print("\nBuilding feature DataFrame...")
    features_df = pd.DataFrame(features_list)
    features_df['label'] = df['label'].values
    
    print(f"\nFeature DataFrame shape: {features_df.shape}")
    print(f"Memory usage: {features_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # ============================================================
    # SANITY CHECK 1: All feature columns present?
    # ============================================================
    print("\n" + "-" * 70)
    print("CHECK 1: Feature columns")
    print("-" * 70)
    missing = set(FEATURE_NAMES) - set(features_df.columns)
    extra = set(features_df.columns) - set(FEATURE_NAMES) - {'label'}
    if missing:
        print(f"  MISSING features: {missing}")
    elif extra:
        print(f"  EXTRA features: {extra}")
    else:
        print(f"  All {len(FEATURE_NAMES)} features present")
    
    # ============================================================
    # SANITY CHECK 2: No NaN values?
    # ============================================================
    print("\n" + "-" * 70)
    print("CHECK 2: NaN values")
    print("-" * 70)
    nan_counts = features_df[FEATURE_NAMES].isna().sum()
    nan_features = nan_counts[nan_counts > 0]
    if len(nan_features) > 0:
        print("  WARNING: NaN found in:")
        print(nan_features)
    else:
        print("  No NaN values in any feature column")
    
    # ============================================================
    # SANITY CHECK 3: Feature statistics by class
    # ============================================================
    print("\n" + "-" * 70)
    print("CHECK 3: Feature averages by class")
    print("-" * 70)
    means_by_class = features_df.groupby('label')[FEATURE_NAMES].mean().round(4)
    print(means_by_class.T)
    
    # ============================================================
    # SANITY CHECK 4: Compare to Week 1 findings
    # ============================================================
    print("\n" + "-" * 70)
    print("CHECK 4: Week 1 comparison")
    print("-" * 70)
    
    week1 = {
        'url_length': {'phish': 46.24, 'legit': 27.23},
        'digit_ratio': {'phish': 0.0636, 'legit': 0.0020},
        'has_https': {'phish': 0.4874, 'legit': 1.0000},
    }
    
    for feat, vals in week1.items():
        phish_actual = features_df[features_df['label']==0][feat].mean()
        legit_actual = features_df[features_df['label']==1][feat].mean()
        print(f"\n  {feat}:")
        print(f"    Phishing  | Week 1: {vals['phish']:.4f} | Now: {phish_actual:.4f} | "
              f"Diff: {abs(phish_actual - vals['phish']):.4f}")
        print(f"    Legit     | Week 1: {vals['legit']:.4f} | Now: {legit_actual:.4f} | "
              f"Diff: {abs(legit_actual - vals['legit']):.4f}")
    
    # ============================================================
    # SANITY CHECK 5: Error analysis
    # ============================================================
    print("\n" + "-" * 70)
    print("CHECK 5: Errors (first 10)")
    print("-" * 70)
    if errors:
        for idx, url, err in errors[:10]:
            print(f"  Row {idx}: {url}")
            print(f"    Error: {err}")
    else:
        print("  Zero errors across all 235k URLs")
    
    # ============================================================
    # SAVE OUTPUT
    # ============================================================
    print("\n" + "-" * 70)
    print("SAVING OUTPUT")
    print("-" * 70)
    output_path = 'data/phiusiil_features.csv'
    features_df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")
    print(f"  Size: {os.path.getsize(output_path) / 1024**2:.2f} MB")
    
    print("\n" + "=" * 70)
    print("INTEGRATION TEST COMPLETE")
    print("=" * 70)
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Throughput: {len(df)/total_time:.0f} URLs/sec")
    print(f"  Errors: {len(errors)} / {len(df)} ({len(errors)/len(df)*100:.4f}%)")
    
    if len(errors) == 0:
        print("\n  ✅ All 24 features work on full dataset. Ready for Week 3.")
    elif len(errors) < 100:
        print(f"\n  🟡 {len(errors)} errors (< 0.1%). Acceptable, investigate later.")
    else:
        print(f"\n  ❌ {len(errors)} errors. Need to fix before Week 3.")


if __name__ == "__main__":
    main()