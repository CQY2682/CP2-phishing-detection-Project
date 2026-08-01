"""
CP2 Hybrid Scorer (Layer 3)
============================

Author: Cheah Qi Yang (22095483)
Module: src/hybrid_scorer.py

Purpose:
    Combines Layer 1 (heuristic engine) + Layer 2 (ML ensemble)
    into a unified phishing verdict with CVSS-style severity.

Architecture:
    1. Run heuristic_check() — if HIGH_RISK, return immediately
    2. Extract 24 features, run XGBoost + RF ensemble
    3. Disagreement Escalation: if heuristic SUSPICIOUS + ML SAFE → ESCALATE
    4. Map final score to CVSS severity

Output:
    {
        'url': str,
        'verdict': 'SAFE' | 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL',
        'score': float (0.0-1.0),
        'cvss': float (0.0-10.0),
        'layer1_verdict': str,
        'layer1_rules': list,
        'layer2_score': float,
        'mitre_techniques': list,
        'escalated': bool,
        'explanation': str,
    }

Usage:
    from src.hybrid_scorer import hybrid_predict
    result = hybrid_predict('https://evil.tk/paypal/login')
"""

import os
import sys
from typing import Any
import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_extractor import extract_features, FEATURE_NAMES
from heuristic_engine import heuristic_check, RULES


# ============================================================
# CONSTANTS
# ============================================================

# Ensemble weights from Milestone A F1 comparison
W_XGB = 0.5001
W_RF  = 0.4999

# Trusted domains — caps ML score to prevent false positives on known legitimate sites
TRUSTED_DOMAINS = {
    'maybank2u.com.my', 'cimbclicks.com.my', 'rhbonlinebanking.com.my',
    'pbebank.com', 'hlb.com.my', 'ambank.com.my', 'myeg.com.my',
    'hasil.gov.my', 'bnm.gov.my', 'grab.com', 'shopee.com.my',
    'lazada.com.my', 'airasia.com', 'malaysiaairlines.com',
    'google.com', 'microsoft.com', 'apple.com', 'github.com',
    'python.org', 'stackoverflow.com', 'wikipedia.org',
    'youtube.com', 'linkedin.com', 'twitter.com', 'facebook.com',
    'amazon.com', 'paypal.com', 'netflix.com', 'spotify.com',
    'sunway.edu.my',
}

def _is_trusted_domain(url: str) -> bool:
    """Check if URL belongs to a trusted domain (subdomain-aware)."""
    try:
        from urllib.parse import urlparse
        netloc = urlparse(url.lower()).netloc.split(':')[0]
        return any(netloc == td or netloc.endswith('.' + td) for td in TRUSTED_DOMAINS)
    except Exception:
        return False

# CVSS-style severity thresholds
# Maps ML score (0.0-1.0) to severity level
SEVERITY_THRESHOLDS = {
    'CRITICAL': 0.90,
    'HIGH':     0.70,
    'MEDIUM':   0.40,
    'LOW':      0.20,
    # Below 0.20 = SAFE
}

# Model paths
XGB_PATH = 'models/xgboost_model.joblib'
RF_PATH  = 'models/rf_model.joblib'


# ============================================================
# MODEL LOADER (lazy load — only loads once)
# ============================================================

_xgb_model: Any = None
_rf_model: Any = None

def _load_models():
    global _xgb_model, _rf_model
    if _xgb_model is None:
        if not os.path.exists(XGB_PATH) or not os.path.exists(RF_PATH):
            raise FileNotFoundError(
                "Models not found. Run python src/model_trainer.py --augmented first."
            )
        _xgb_model = joblib.load(XGB_PATH)
        _rf_model  = joblib.load(RF_PATH)
    return _xgb_model, _rf_model


# ============================================================
# SEVERITY MAPPER
# ============================================================

def score_to_severity(score: float) -> str:
    """Map 0.0-1.0 phishing score to CVSS-style severity label."""
    if score >= SEVERITY_THRESHOLDS['CRITICAL']:
        return 'CRITICAL'
    elif score >= SEVERITY_THRESHOLDS['HIGH']:
        return 'HIGH'
    elif score >= SEVERITY_THRESHOLDS['MEDIUM']:
        return 'MEDIUM'
    elif score >= SEVERITY_THRESHOLDS['LOW']:
        return 'LOW'
    else:
        return 'SAFE'


def score_to_cvss(score: float) -> float:
    """Map 0.0-1.0 score to CVSS 0.0-10.0 scale."""
    return round(score * 10, 1)


def build_explanation(
    url, layer1_verdict, layer1_rules,
    layer2_score, final_score, verdict, escalated
) -> str:
    """Build human-readable explanation of verdict."""
    parts = []

    if layer1_verdict == 'HIGH_RISK':
        rule_descs = []
        for r in layer1_rules:
            if r in RULES:
                _, _, mitre, desc = RULES[r]
                rule_descs.append(f"{desc} ({', '.join(mitre)})")
        parts.append(f"Heuristic engine flagged HIGH_RISK: {'; '.join(rule_descs)}.")
        parts.append("ML ensemble bypassed.")

    elif layer1_verdict == 'SUSPICIOUS' and escalated:
        parts.append(
            f"Heuristic engine flagged SUSPICIOUS "
            f"(rules: {', '.join(layer1_rules)}) "
            f"but ML ensemble scored LOW ({layer2_score:.3f}). "
            f"Disagreement Escalation applied — verdict elevated to {verdict}."
        )

    else:
        parts.append(
            f"ML ensemble score: {layer2_score:.3f} "
            f"(XGBoost × {W_XGB} + RandomForest × {W_RF})."
        )
        if layer1_verdict == 'SUSPICIOUS':
            parts.append(
                f"Heuristic flagged SUSPICIOUS "
                f"({', '.join(layer1_rules)}) — combined with ML score."
            )
        if verdict == 'SAFE':
            parts.append("No phishing indicators detected.")

    return ' '.join(parts)


# ============================================================
# MAIN PREDICT FUNCTION
# ============================================================

def hybrid_predict(url: str) -> dict:
    """
    Full hybrid prediction pipeline for a single URL.

    Args:
        url: Raw URL string

    Returns:
        dict with verdict, score, cvss, explanation, and metadata
    """
    if not isinstance(url, str) or not url.strip():
        return {
            'url': url,
            'verdict': 'SAFE',
            'score': 0.0,
            'cvss': 0.0,
            'layer1_verdict': 'CLEAN',
            'layer1_rules': [],
            'layer2_score': 0.0,
            'mitre_techniques': [],
            'escalated': False,
            'explanation': 'Empty or invalid URL.',
        }

    # ---- LAYER 1: Heuristic Engine ----
    layer1_verdict, layer1_rules, mitre_techniques = heuristic_check(url)

    if layer1_verdict == 'HIGH_RISK':
        # Instant flag — bypass ML entirely
        return {
            'url': url,
            'verdict': 'CRITICAL',
            'score': 1.0,
            'cvss': 10.0,
            'layer1_verdict': layer1_verdict,
            'layer1_rules': layer1_rules,
            'layer2_score': None,
            'mitre_techniques': mitre_techniques,
            'escalated': False,
            'explanation': build_explanation(
                url, layer1_verdict, layer1_rules,
                None, 1.0, 'CRITICAL', False
            ),
        }

    # ---- LAYER 2: ML Ensemble ----
    xgb_model, rf_model = _load_models()
    features = extract_features(url)
    X = pd.DataFrame([features])[FEATURE_NAMES]

    xgb_proba = xgb_model.predict_proba(X)[0][1]
    rf_proba  = rf_model.predict_proba(X)[0][1]
    layer2_score = W_XGB * xgb_proba + W_RF * rf_proba

    # ---- LAYER 3: Disagreement Escalation ----
    escalated = False
    final_score = layer2_score

    # Trusted domain cap — prevents false positives on known legitimate sites
    if _is_trusted_domain(url) and layer2_score > 0.3:
        final_score = 0.15  # cap at SAFE threshold

    if layer1_verdict == 'SUSPICIOUS' and layer2_score < 0.5:
        # Heuristic says suspicious, ML says safe → ESCALATE
        final_score = max(layer2_score, 0.55)
        escalated = True
        # Add heuristic MITRE techniques to output
        for r in layer1_rules:
            if r in RULES:
                _, _, mitre, _ = RULES[r]
                for t in mitre:
                    if t not in mitre_techniques:
                        mitre_techniques.append(t)

    elif layer1_verdict == 'SUSPICIOUS':
        # Both agree suspicious — boost score slightly
        final_score = min(layer2_score + 0.05, 1.0)

    verdict = score_to_severity(final_score)

    return {
        'url': url,
        'verdict': verdict,
        'score': round(final_score, 4),
        'cvss': score_to_cvss(final_score),
        'layer1_verdict': layer1_verdict,
        'layer1_rules': layer1_rules,
        'layer2_score': round(layer2_score, 4),
        'mitre_techniques': mitre_techniques,
        'escalated': escalated,
        'explanation': build_explanation(
            url, layer1_verdict, layer1_rules,
            layer2_score, final_score, verdict, escalated
        ),
    }


# ============================================================
# SELF-TEST
# ============================================================

if __name__ == "__main__":
    test_urls = [
        # HIGH_RISK (heuristic should instant-flag)
        ("https://xn--pypal-4ve.com/login",          "Punycode → CRITICAL"),
        ("http://192.168.1.1/login",                  "IP+login → CRITICAL"),
        ("https://evil.com%25252Flogin",               "Triple-encoded → CRITICAL"),
        ("https://m365-login.suspicious.tk/auth",     "AiTM → CRITICAL"),
        # Phishing caught by ML
        ("https://paypal-secure-login.evil.tk/account/verify?id=12345", "Phishing → HIGH/CRITICAL"),
        ("https://secure-bank-update.xyz/login?ref=email", "Phishing → HIGH"),
        # Legitimate
        ("https://www.google.com",                    "Legit → SAFE"),
        ("https://www.maybank2u.com.my/login",        "Real bank → LOW/SAFE"),
        ("https://docs.python.org/3/library/urllib.parse.html", "Legit path → depends"),
    ]

    print("=" * 75)
    print("HYBRID SCORER — Self-Test")
    print("=" * 75)

    _load_models()
    print("Models loaded.\n")

    for url, expected in test_urls:
        result = hybrid_predict(url)
        print(f"URL: {url[:70]}")
        print(f"Expected:  {expected}")
        print(f"Verdict:   {result['verdict']} (score={result['score']}, cvss={result['cvss']})")
        print(f"Layer 1:   {result['layer1_verdict']} {result['layer1_rules']}")
        print(f"Layer 2:   {result['layer2_score']}")
        print(f"Escalated: {result['escalated']}")
        print(f"MITRE:     {result['mitre_techniques']}")
        print(f"Why:       {result['explanation'][:120]}")
        print("-" * 75)