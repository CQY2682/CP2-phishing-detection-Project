"""
Model Bias Stress Test
======================

Tests trained models against URLs that break PhiUSIIL's known biases:
1. Legitimate URLs with long paths (bias: legit = homepage only)
2. Legitimate URLs with many slashes
3. Legitimate HTTP URLs (bias: legit = 100% HTTPS)
4. Real phishing URLs that use HTTPS (modern phishing)

Run: python tests/test_model_bias.py
"""

import sys
import os
import pandas as pd
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from feature_extractor import extract_features, FEATURE_NAMES

# Load models
xgb_model = joblib.load('models/xgboost_model.joblib')
rf_model = joblib.load('models/rf_model.joblib')

# Ensemble weights from training
W_XGB = 0.5
W_RF = 0.5

def predict(url):
    features = extract_features(url)
    X = pd.DataFrame([features])[FEATURE_NAMES]
    xgb_p = xgb_model.predict_proba(X)[0][1]
    rf_p = rf_model.predict_proba(X)[0][1]
    score = W_XGB * xgb_p + W_RF * rf_p
    verdict = "PHISHING" if score >= 0.5 else "LEGIT"
    return verdict, score, features

test_urls = [
    # ---- SHOULD BE LEGIT (model might wrongly flag) ----
    ("https://docs.python.org/3/library/urllib.parse.html",
     "LEGIT", "Long path, many slashes, legit"),

    ("https://github.com/CQY2682/CP2-phishing-detection-Project/blob/main/src/feature_extractor.py",
     "LEGIT", "Very long path, many slashes, legit"),

    ("https://www.stackoverflow.com/questions/12345678/how-to-parse-urls-in-python",
     "LEGIT", "Long path, legit"),

    ("http://neverssl.com/",
     "LEGIT", "HTTP only (no HTTPS), legitimate site"),

    ("http://www.example.com/login",
     "LEGIT", "HTTP + login keyword, should still be legit"),

    ("https://accounts.google.com/signin/v2/identifier?flowName=GlifWebSignIn",
     "LEGIT", "Real Google signin URL — many params"),

    ("https://www.amazon.com/gp/cart/view.html?ref_=nav_cart",
     "LEGIT", "Real Amazon URL with path"),

    # ---- SHOULD BE PHISHING ----
    ("https://paypal-secure-login.evil.tk/account/verify?id=12345",
     "PHISHING", "Classic phishing"),

    ("https://www.paypa1.com/login",
     "PHISHING", "Typosquat phishing"),

    ("http://192.168.1.1/admin/login",
     "PHISHING", "IP-based phishing"),

    ("https://secure-bank-update.xyz/customer/login?ref=email",
     "PHISHING", "Keyword-heavy phishing"),

    ("https://bit.ly/3xK9pQz",
     "PHISHING", "URL shortener (hides destination)"),
]

print("=" * 75)
print("MODEL BIAS STRESS TEST")
print("=" * 75)
print(f"\n{'URL':<55} {'Expected':<10} {'Got':<10} {'Score':<6} {'Pass?'}")
print("-" * 75)

passed = 0
failed = 0
false_positives = []
false_negatives = []

for url, expected, desc in test_urls:
    verdict, score, features = predict(url)
    correct = verdict == expected
    status = "✅" if correct else "❌"
    if correct:
        passed += 1
    else:
        failed += 1
        if expected == "LEGIT":
            false_positives.append((url, score, desc))
        else:
            false_negatives.append((url, score, desc))
    
    short_url = url[:52] + "..." if len(url) > 52 else url
    print(f"{short_url:<55} {expected:<10} {verdict:<10} {score:.3f}  {status}")

print("-" * 75)
print(f"\nResults: {passed}/{len(test_urls)} passed")

if false_positives:
    print(f"\n⚠️  FALSE POSITIVES (legit URLs flagged as phishing):")
    for url, score, desc in false_positives:
        print(f"  Score {score:.3f} | {desc}")
        print(f"  URL: {url[:80]}")
        # Show which features likely caused the problem
        features = extract_features(url)
        print(f"  qty_slash={features['qty_slash']} | has_https={features['has_https']} | url_length={features['url_length']}")

if false_negatives:
    print(f"\n⚠️  FALSE NEGATIVES (phishing URLs missed):")
    for url, score, desc in false_negatives:
        print(f"  Score {score:.3f} | {desc}")

print("\n" + "=" * 75)
print("BIAS ANALYSIS")
print("=" * 75)
print("\nKey features driving model:")
print("  qty_slash: 50.21% importance")
print("  has_https: 48.81% importance")
print("\nPhiUSIIL bias:")
print("  Legitimate = homepage only (max 2 slashes, always HTTPS)")
print("  Phishing = paths + params (3+ slashes, 48% HTTP)")
print("\nStress test checks if model works BEYOND this bias.")