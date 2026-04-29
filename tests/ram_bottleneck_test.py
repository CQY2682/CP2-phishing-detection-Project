"""
RAM Bottleneck Test for CP2 Project
Purpose: Empirically demonstrate 16GB insufficient for full ML pipeline
NOT for production CP2 use — uses simplified features only
"""

import os
import time
import psutil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib
from urllib.parse import urlparse


def log_memory(stage_name):
    """Log current RAM usage with timestamp."""
    process = psutil.Process(os.getpid())
    process_mb = process.memory_info().rss / (1024 ** 2)
    system = psutil.virtual_memory()
    system_used_gb = (system.total - system.available) / (1024 ** 3)
    system_total_gb = system.total / (1024 ** 3)
    timestamp = time.strftime("%H:%M:%S")
    print(f"\n[{timestamp}] === {stage_name} ===")
    print(f"  Python process: {process_mb:.1f} MB")
    print(f"  System total used: {system_used_gb:.2f} / {system_total_gb:.2f} GB "
          f"({system_used_gb/system_total_gb*100:.1f}%)")
    print(f"  Available: {system.available / (1024**3):.2f} GB")


def extract_simple_features(url):
    """Extract 10 simple lexical features. Quick subset for RAM test only."""
    try:
        parsed = urlparse(url)
        return {
            'url_length': len(url),
            'domain_length': len(parsed.netloc),
            'path_length': len(parsed.path),
            'num_dots': url.count('.'),
            'num_hyphens': url.count('-'),
            'num_slashes': url.count('/'),
            'num_digits': sum(c.isdigit() for c in url),
            'has_https': 1 if url.startswith('https') else 0,
            'has_at': 1 if '@' in url else 0,
            'has_ip': 1 if any(c.isdigit() for c in parsed.netloc.split('.')[0:1]) else 0,
        }
    except Exception:
        return {k: 0 for k in ['url_length', 'domain_length', 'path_length',
                                'num_dots', 'num_hyphens', 'num_slashes',
                                'num_digits', 'has_https', 'has_at', 'has_ip']}


# ============================================================
# STAGE A — Load PhiUSIIL
# ============================================================
log_memory("STAGE A — BEFORE LOADING CSV")
print("Loading PhiUSIIL dataset...")

df = pd.read_csv('data/PhiUSIIL_Phishing_URL_Dataset.csv')
print(f"Loaded {len(df):,} rows × {len(df.columns)} columns")

log_memory("STAGE A — AFTER LOADING CSV")

# ============================================================
# STAGE B — Feature extraction (URL only, ignore 54 pre-extracted)
# ============================================================
print("\nExtracting features from raw URLs (this takes 30-60 sec)...")
features_list = []
for i, url in enumerate(df['URL']):
    if i % 50000 == 0 and i > 0:
        print(f"  Processed {i:,} URLs...")
    features_list.append(extract_simple_features(url))

X = pd.DataFrame(features_list)
y = df['label']

log_memory("STAGE B — AFTER FEATURE EXTRACTION")

# Free memory we don't need
del df, features_list
import gc
gc.collect()

# ============================================================
# STAGE C — Train/test split
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"\nTrain: {len(X_train):,} rows | Test: {len(X_test):,} rows")
log_memory("STAGE C — AFTER TRAIN/TEST SPLIT")

# ============================================================
# STAGE D — Train XGBoost
# ============================================================
print("\nTraining XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    n_jobs=-1,
    random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)
joblib.dump(xgb_model, 'models/xgboost_test.joblib')
print(f"XGBoost test accuracy: {xgb_model.score(X_test, y_test):.4f}")
log_memory("STAGE D — AFTER XGBOOST TRAINING")

# ============================================================
# STAGE E — Train Random Forest
# ============================================================
print("\nTraining Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    n_jobs=-1,
    random_state=42
)
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, 'models/rf_test.joblib')
print(f"Random Forest test accuracy: {rf_model.score(X_test, y_test):.4f}")
log_memory("STAGE E — AFTER RANDOM FOREST TRAINING")

# ============================================================
# STAGE F — SHAP TreeExplainer (THE KILLER)
# ============================================================
print("\n" + "="*60)
print("STAGE F: SHAP analysis on FULL test set (47,159 rows)")
print("This is the stage most likely to OOM crash on 16GB.")
print("If system freezes, take phone photo of screen.")
print("="*60)

import shap

# XGBoost SHAP (usually faster)
print("\nComputing SHAP values for XGBoost on full test set...")
log_memory("STAGE F — BEFORE SHAP (XGBoost)")
xgb_explainer = shap.TreeExplainer(xgb_model)
xgb_shap_values = xgb_explainer.shap_values(X_test)
log_memory("STAGE F — AFTER SHAP (XGBoost)")

# Random Forest SHAP (this is the worst killer)
print("\nComputing SHAP values for Random Forest on full test set...")
print("⚠️  This is the heaviest stage. Watch Task Manager.")
log_memory("STAGE F — BEFORE SHAP (Random Forest)")
rf_explainer = shap.TreeExplainer(rf_model)
rf_shap_values = rf_explainer.shap_values(X_test)
log_memory("STAGE F — AFTER SHAP (Random Forest)")

print("\n" + "="*60)
print("✅ ALL STAGES COMPLETED (if you're seeing this, system survived)")
print("="*60)