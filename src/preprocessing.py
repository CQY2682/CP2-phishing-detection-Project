"""
CP2 Preprocessing Pipeline
==========================

Author: Cheah Qi Yang (22095483)
Module: src/preprocessing.py

Purpose:
    Transform raw PhiUSIIL dataset into ML-ready train/test splits.
    
Steps:
    1. Load raw CSV
    2. Keep only URL + label columns (ignore 54 pre-extracted features)
    3. Remove 425 duplicate URLs
    4. Flip labels: 1=phishing, 0=legitimate (ML convention)
    5. Extract 24 features per URL
    6. Stratified 80/20 train/test split (random_state=42)
    7. Save splits as separate files

CP2 Critical Rule:
    Use only URL and label columns from PhiUSIIL.
    Ignore the other 54 pre-extracted columns.

Usage:
    python src/preprocessing.py
    
Outputs (in data/ folder):
    - X_train.csv  (188,296 × 24)
    - X_test.csv   (47,074 × 24)
    - y_train.csv  (188,296 × 1)
    - y_test.csv   (47,074 × 1)
"""

import os
import sys
import time
import pandas as pd
from sklearn.model_selection import train_test_split

# Add src to path so we can import feature_extractor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_extractor import extract_features, FEATURE_NAMES


# Constants
RAW_DATA_PATH = 'data/PhiUSIIL_Phishing_URL_Dataset.csv'
OUTPUT_DIR = 'data'
RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_raw_dataset(path: str) -> pd.DataFrame:
    """Load raw PhiUSIIL CSV, keep only URL + label columns."""
    print(f"\n[1/6] Loading raw dataset: {path}")
    df = pd.read_csv(path, usecols=['URL', 'label'])
    print(f"  Loaded {len(df):,} rows × {len(df.columns)} columns (URL + label only)")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate URLs (425 expected in PhiUSIIL)."""
    print("\n[2/6] Removing duplicate URLs...")
    before = len(df)
    df = df.drop_duplicates(subset=['URL'], keep='first').reset_index(drop=True)
    removed = before - len(df)
    print(f"  Removed {removed} duplicates -> {len(df):,} unique URLs remain")
    return df


def flip_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flip PhiUSIIL labels to ML convention.
    
    PhiUSIIL native:   0 = phishing, 1 = legitimate (REVERSED)
    After flip:        1 = phishing, 0 = legitimate (standard)
    
    This makes precision/recall/F1 scoring intuitive (phishing = positive class).
    """
    print("\n[3/6] Flipping labels to ML convention...")
    print(f"  Before: {df['label'].value_counts().to_dict()}")
    print("  Original PhiUSIIL: 0=phishing, 1=legitimate")
    
    df['label'] = 1 - df['label']
    
    print(f"  After:  {df['label'].value_counts().to_dict()}")
    print("  ML convention: 1=phishing, 0=legitimate")
    return df


def extract_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply extract_features() to all URLs. Returns DataFrame of 24 features."""
    print(f"\n[4/6] Extracting {len(FEATURE_NAMES)} features from {len(df):,} URLs...")
    start = time.time()
    
    features_list = []
    for i, url in enumerate(df['URL']):
        features_list.append(extract_features(url))
        if (i + 1) % 50000 == 0:
            elapsed = time.time() - start
            print(f"  [{i+1:>6,}/{len(df):,}] {elapsed:.1f}s, "
                  f"{(i+1)/elapsed:.0f} URLs/sec")
    
    X = pd.DataFrame(features_list)
    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s ({len(df)/elapsed:.0f} URLs/sec)")
    print(f"  Feature matrix shape: {X.shape}")
    return X


def split_train_test(X: pd.DataFrame, y: pd.Series):
    """Stratified 80/20 train/test split with random_state=42."""
    print(f"\n[5/6] Splitting train/test (80/20, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )
    print(f"  Train: {len(X_train):,} rows ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Test:  {len(X_test):,} rows ({len(X_test)/len(X)*100:.1f}%)")
    print(f"\n  Train label distribution:")
    print(f"    Phishing  (1): {(y_train==1).sum():,} ({(y_train==1).mean()*100:.2f}%)")
    print(f"    Legitimate (0): {(y_train==0).sum():,} ({(y_train==0).mean()*100:.2f}%)")
    print(f"\n  Test label distribution:")
    print(f"    Phishing  (1): {(y_test==1).sum():,} ({(y_test==1).mean()*100:.2f}%)")
    print(f"    Legitimate (0): {(y_test==0).sum():,} ({(y_test==0).mean()*100:.2f}%)")
    return X_train, X_test, y_train, y_test


def save_splits(X_train, X_test, y_train, y_test, output_dir: str):
    """Save 4 files: X_train, X_test, y_train, y_test."""
    print(f"\n[6/6] Saving splits to {output_dir}/...")
    
    files = [
        (X_train, 'X_train.csv'),
        (X_test, 'X_test.csv'),
        (y_train.rename('label').to_frame(), 'y_train.csv'),
        (y_test.rename('label').to_frame(), 'y_test.csv'),
    ]
    
    for data, filename in files:
        path = os.path.join(output_dir, filename)
        data.to_csv(path, index=False)
        size_mb = os.path.getsize(path) / 1024**2
        print(f"  {filename:15s} -> {data.shape}, {size_mb:.2f} MB")


def main():
    print("=" * 70)
    print("CP2 PREPROCESSING PIPELINE")
    print("=" * 70)
    total_start = time.time()
    
    # Verify input exists
    if not os.path.exists(RAW_DATA_PATH):
        print(f"\nERROR: Raw dataset not found at {RAW_DATA_PATH}")
        print("Make sure PhiUSIIL_Phishing_URL_Dataset.csv is in data/")
        sys.exit(1)
    
    # Pipeline
    df = load_raw_dataset(RAW_DATA_PATH)
    df = remove_duplicates(df)
    df = flip_labels(df)
    
    X = extract_all_features(df)
    y = df['label']
    
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    save_splits(X_train, X_test, y_train, y_test, OUTPUT_DIR)
    
    # Summary
    total_time = time.time() - total_start
    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Output files in: {OUTPUT_DIR}/")
    print(f"  Ready for: Week 4 model training")


if __name__ == "__main__":
    main()