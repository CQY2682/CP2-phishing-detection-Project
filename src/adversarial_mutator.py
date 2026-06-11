"""
CP2 Adversarial Robustness Testing (Milestone D)
Author: Cheah Qi Yang (22095483)
"""

import os
import sys
import random
import urllib.parse
import json
from typing import Any
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_extractor import extract_features, FEATURE_NAMES
from heuristic_engine import heuristic_check

# Load ensemble weights from training metrics
with open('reports/training_metrics.json') as f:
    _m = json.load(f)
W_XGB = _m['ensemble_weights']['xgboost_weight']
W_RF  = _m['ensemble_weights']['rf_weight']

_xgb: Any = None
_rf:  Any = None

def _load():
    global _xgb, _rf
    if _xgb is None:
        _xgb = joblib.load('models/xgboost_model.joblib')
        _rf  = joblib.load('models/rf_model.joblib')
    return _xgb, _rf

def ml_score(url):
    xgb, rf = _load()
    X = pd.DataFrame([extract_features(url)])[FEATURE_NAMES]
    return W_XGB * xgb.predict_proba(X)[0][1] + W_RF * rf.predict_proba(X)[0][1]

def is_detected(url):
    h, _, _ = heuristic_check(url)
    return h == 'HIGH_RISK' or ml_score(url) >= 0.5

# ── Mutations ──────────────────────────────────────────────

BENIGN_SUBS  = ['www','secure','login','account','support','portal']
BENIGN_PATHS = ['/legal','/privacy','/terms','/about','/help']

def mut_subdomain(url):
    p = urllib.parse.urlparse(url)
    netloc = p.netloc[4:] if p.netloc.startswith('www.') else p.netloc
    return p._replace(netloc=f"{random.choice(BENIGN_SUBS)}.{netloc}").geturl()

def mut_https(url):
    p = urllib.parse.urlparse(url)
    return p._replace(scheme='https').geturl()

def mut_path_dilution(url):
    p = urllib.parse.urlparse(url)
    extra = '/'.join(random.sample(BENIGN_PATHS, k=2))
    new_path = (p.path.rstrip('/') or '') + '/' + extra
    return p._replace(path=new_path).geturl()

def mut_keyword_removal(url):
    p = urllib.parse.urlparse(url)
    path = p.path
    for kw in ['login','verify','account','secure','update','confirm','auth']:
        path = path.replace(kw, 'page')
    return p._replace(path=path).geturl()

def mut_combined(url):
    for fn in [mut_subdomain, mut_path_dilution, mut_keyword_removal]:
        if random.random() > 0.4:
            url = fn(url)
    return url

MUTATORS = {
    'subdomain_padding':  mut_subdomain,
    'https_laundering':   mut_https,
    'path_dilution':      mut_path_dilution,
    'keyword_removal':    mut_keyword_removal,
    'combined':           mut_combined,
}

# ── Main ───────────────────────────────────────────────────

PHISHING_URLS = [
    "http://paypal-secure-login.evil.tk/account/verify?id=12345",
    "http://m365-update.suspicious.cf/auth/login",
    "http://192.168.1.1/admin/login",
    "http://banking-update.secure-login.ml/confirm",
    "http://appleid-locked.tk/verify/account?ref=email",
    "http://microsoft-365-login.gq/signin",
    "http://secure-bank-update.xyz/customer/login",
    "http://paypa1.com/login/verify",
    "http://wellsfarg0.com/account/secure",
    "http://cimb-clicks-login.evil.tk/user/verify",
    "http://maybank-secure.cf/login/confirm?id=999",
    "http://grab-support-update.ml/account/verify",
    "http://netflix-billing-update.gq/payment/confirm",
    "http://amazon-prize-claim.tk/gifts/redeem?ref=123",
    "http://steamcommunity-trade.evil.xyz/tradeoffer/new",
    "http://instagram-verify.gq/accounts/verify",
    "http://whatsapp-prize.cf/claim/winner?id=abc",
    "http://dropbox-secure.ml/signin/verify",
    "http://office365-update.gq/auth/mfa/login",
    "http://hsbc-online-banking.tk/login/secure",
]

def main():
    print("=" * 60)
    print("CP2 ADVERSARIAL ROBUSTNESS TEST (Milestone D)")
    print("=" * 60)

    results = []
    total = len(PHISHING_URLS)

    for i, url in enumerate(PHISHING_URLS):
        score_before = ml_score(url)
        detected_before = is_detected(url)
        h_before, _, _ = heuristic_check(url)

        for mut_name, mut_fn in MUTATORS.items():
            mutated = mut_fn(url)
            score_after = ml_score(mutated)
            h_after, _, _ = heuristic_check(mutated)
            detected_after = is_detected(mutated)

            results.append({
                'original_url':     url,
                'mutated_url':      mutated,
                'mutation':         mut_name,
                'score_before':     round(score_before, 4),
                'score_after':      round(score_after, 4),
                'score_drop':       round(score_before - score_after, 4),
                'heuristic_before': h_before,
                'heuristic_after':  h_after,
                'detected_before':  detected_before,
                'detected_after':   detected_after,
                'evaded':           detected_before and not detected_after,
            })

        if (i+1) % 5 == 0:
            print(f"  Processed {i+1}/{total} URLs...")

    df = pd.DataFrame(results)
    os.makedirs('reports', exist_ok=True)
    df.to_csv('reports/adversarial_results.csv', index=False)

    # ── Summary ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY BY MUTATION TYPE")
    print("=" * 60)
    summary = df.groupby('mutation').agg(
        avg_score_drop  = ('score_drop', 'mean'),
        detection_rate  = ('detected_after', 'mean'),
        evasion_rate    = ('evaded', 'mean'),
    ).round(4)
    print(summary)

    print("\n" + "=" * 60)
    print("OVERALL STATS")
    print("=" * 60)
    print(f"  URLs tested:       {total}")
    print(f"  Mutations applied: {len(df)}")
    print(f"  Overall detection: {df['detected_after'].mean()*100:.1f}%")
    print(f"  Overall evasion:   {df['evaded'].mean()*100:.1f}%")
    print("\n  Saved: reports/adversarial_results.csv")
    print("\n[DONE] Milestone D complete.")

if __name__ == "__main__":
    main()