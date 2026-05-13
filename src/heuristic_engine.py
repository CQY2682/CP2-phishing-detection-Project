"""
CP2 Heuristic Engine (Layer 1)
==============================

Author: Cheah Qi Yang (22095483)
Module: src/heuristic_engine.py

Purpose:
    Rule-based instant phishing flagger. Layer 1 of 3-layer architecture.
    Catches "smoking gun" patterns where ML overhead is unnecessary.
    
Architecture Role:
    URL → HEURISTIC ENGINE → if HIGH_RISK return immediately
                          → else pass to ML ensemble (Layer 2)

Output:
    Tuple of (verdict, triggered_rules, mitre_techniques)
    
    verdict: 'HIGH_RISK', 'SUSPICIOUS', or 'CLEAN'
    triggered_rules: list of rule names that fired
    mitre_techniques: list of MITRE ATT&CK technique IDs

Usage:
    from src.heuristic_engine import heuristic_check
    result = heuristic_check("https://xn--pypal.com/login")
    # result = ('HIGH_RISK', ['idn_punycode'], ['T1036.007'])
"""

import re
import sys
import os
from typing import Tuple, List, Dict
from urllib.parse import urlparse

# Import feature functions we reuse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_extractor import (
    is_ip,
    idn_homograph_flag,
    recursive_decode_depth,
    has_login_keyword,
    brand_in_path_not_domain,
    tld_risk_score,
    levenshtein_min,
    qty_at,
)


# ============================================================
# RULE PARAMETERS (TUNABLE)
# ============================================================

DECODE_DEPTH_THRESHOLD = 2          # Rule 3
TLD_RISK_HIGH_THRESHOLD = 0.9       # Rule 5
LEVENSHTEIN_TYPOSQUAT = 1           # Rule 8

# AiTM (Adversary-in-the-Middle) lure patterns
# These are commodity phishing kits in 2026 (post-Tycoon-2FA takedown)
AITM_PATTERNS = [
    r'm365[-_]?login',
    r'office365[-_]?secure',
    r'microsoft[-_]?online',
    r'outlook[-_]?security',
    r'azure[-_]?ad[-_]?login',
    r'sso[-_]?mail',
    r'webmail[-_]?login',
    r'okta[-_]?verify',
    r'duo[-_]?security',
    r'gmail[-_]?signin',
    r'apple[-_]?id[-_]?signin',
]
AITM_REGEX = re.compile('|'.join(AITM_PATTERNS), re.IGNORECASE)


# ============================================================
# INDIVIDUAL RULES
# ============================================================

def rule_ip_url(url: str) -> bool:
    """Rule 1: URL uses IP address instead of domain. MITRE T1071.001"""
    return is_ip(url) == 1


def rule_punycode(url: str) -> bool:
    """Rule 2: URL contains Punycode (IDN homograph attack). MITRE T1036.007"""
    return idn_homograph_flag(url) == 1


def rule_multi_encoded(url: str) -> bool:
    """Rule 3: URL is multi-layer encoded (suspicious obfuscation). MITRE T1027.001"""
    return recursive_decode_depth(url) >= DECODE_DEPTH_THRESHOLD


def rule_aitm_pattern(url: str) -> bool:
    """Rule 4: URL matches known AiTM phishing kit patterns. MITRE T1557"""
    return bool(AITM_REGEX.search(url))


def rule_brand_spoof_sus_tld(url: str) -> bool:
    """
    Rule 5: Brand name in path + URL on suspicious TLD.
    Classic phishing kit pattern (e.g., paypal in /paypal/login on .tk).
    MITRE T1583.001 + T1566.002
    """
    return (brand_in_path_not_domain(url) == 1 and
            tld_risk_score(url) >= TLD_RISK_HIGH_THRESHOLD)


def rule_ip_with_login(url: str) -> bool:
    """
    Rule 6: IP-based URL containing login keywords.
    Legitimate sites use domain names, not IPs, for login pages.
    MITRE T1071.001 + T1566.002
    """
    return is_ip(url) == 1 and has_login_keyword(url) == 1


def rule_embedded_credentials(url: str) -> bool:
    """
    Rule 7: URL contains '@' character (credential embedding attack).
    Example: http://legitimate.com@evil.com tricks user with @-spoof.
    MITRE T1566.002
    """
    return qty_at(url) >= 1


def rule_typosquat(url: str) -> bool:
    """
    Rule 8: Levenshtein distance to known brand == 1.
    Catches paypa1.com, googel.com, micr0soft.com, etc.
    MITRE T1566.002
    """
    return levenshtein_min(url) == LEVENSHTEIN_TYPOSQUAT


# ============================================================
# RULE REGISTRY
# ============================================================

# Maps: rule_name -> (rule_function, severity, mitre_techniques, description)
RULES: Dict[str, Tuple] = {
    'ip_url': (
        rule_ip_url, 'HIGH_RISK', ['T1071.001'],
        'URL uses raw IP address instead of domain name'
    ),
    'idn_punycode': (
        rule_punycode, 'HIGH_RISK', ['T1036.007'],
        'URL contains Punycode (xn--) — IDN homograph attack'
    ),
    'multi_encoded': (
        rule_multi_encoded, 'HIGH_RISK', ['T1027.001'],
        f'URL encoded {DECODE_DEPTH_THRESHOLD}+ times (multi-layer obfuscation)'
    ),
    'aitm_pattern': (
        rule_aitm_pattern, 'HIGH_RISK', ['T1557'],
        'URL matches known AiTM phishing kit pattern'
    ),
    'brand_spoof_sus_tld': (
        rule_brand_spoof_sus_tld, 'HIGH_RISK', ['T1583.001', 'T1566.002'],
        'Brand name in path AND URL on suspicious TLD'
    ),
    'ip_with_login': (
        rule_ip_with_login, 'HIGH_RISK', ['T1071.001', 'T1566.002'],
        'IP-based URL with login/auth keywords'
    ),
    'embedded_credentials': (
        rule_embedded_credentials, 'SUSPICIOUS', ['T1566.002'],
        'URL contains @ character (credential embedding pattern)'
    ),
    'typosquat': (
        rule_typosquat, 'HIGH_RISK', ['T1566.002'],
        'Domain Levenshtein distance == 1 from known brand'
    ),
}


# ============================================================
# MAIN HEURISTIC FUNCTION
# ============================================================

def heuristic_check(url: str) -> Tuple[str, List[str], List[str]]:
    """
    Run all heuristic rules against a URL.
    
    Returns:
        (verdict, triggered_rules, mitre_techniques)
        
        verdict: 'HIGH_RISK' if any HIGH_RISK rule triggers,
                 'SUSPICIOUS' if only SUSPICIOUS rules trigger,
                 'CLEAN' if no rules trigger.
        triggered_rules: list of rule names that fired
        mitre_techniques: deduplicated list of MITRE technique IDs
    
    Note:
        'CLEAN' verdict means the heuristic engine found nothing.
        The URL is then passed to the ML ensemble (Layer 2).
        It does NOT mean the URL is safe.
    """
    if not isinstance(url, str) or not url:
        return 'CLEAN', [], []
    
    triggered = []
    techniques = []
    has_high_risk = False
    
    for rule_name, (rule_fn, severity, mitre, _desc) in RULES.items():
        try:
            if rule_fn(url):
                triggered.append(rule_name)
                techniques.extend(mitre)
                if severity == 'HIGH_RISK':
                    has_high_risk = True
        except Exception:
            # Rule errors should not crash the engine
            continue
    
    # Deduplicate MITRE techniques while preserving order
    seen = set()
    techniques = [t for t in techniques if not (t in seen or seen.add(t))]
    
    if has_high_risk:
        verdict = 'HIGH_RISK'
    elif triggered:
        verdict = 'SUSPICIOUS'
    else:
        verdict = 'CLEAN'
    
    return verdict, triggered, techniques


def explain_verdict(url: str) -> str:
    """Pretty-print heuristic decision for one URL (debugging/dashboard use)."""
    verdict, rules, techniques = heuristic_check(url)
    lines = [
        f"URL: {url[:100]}",
        f"Verdict: {verdict}",
    ]
    if rules:
        lines.append("Triggered rules:")
        for r in rules:
            _, severity, mitre, desc = RULES[r]
            lines.append(f"  - [{severity}] {r}: {desc} ({', '.join(mitre)})")
    else:
        lines.append("No rules triggered (passes to ML ensemble)")
    return '\n'.join(lines)


# ============================================================
# SELF-TEST
# ============================================================

if __name__ == "__main__":
    test_urls = [
        # Expected HIGH_RISK
        ("http://192.168.1.1/login", "IP + login → HIGH_RISK"),
        ("https://xn--pypal-4ve.com/login", "Punycode → HIGH_RISK"),
        ("https://evil.com%25252Flogin", "Triple-encoded → HIGH_RISK"),
        ("https://m365-login.suspicious.tk/auth", "AiTM pattern → HIGH_RISK"),
        ("https://evil.tk/paypal/login", "Brand spoof + sus TLD → HIGH_RISK"),
        ("https://paypa1.com/login", "Typosquat → HIGH_RISK"),
        # Expected SUSPICIOUS
        ("http://legitimate.com@evil.com/path", "@ credential embed → SUSPICIOUS"),
        # Expected CLEAN
        ("https://www.google.com", "Legitimate → CLEAN"),
        ("https://docs.python.org/3/library/urllib.parse.html", "Long legit → CLEAN"),
        ("https://www.paypal.com", "Brand exact match → CLEAN"),
    ]
    
    print("=" * 70)
    print("HEURISTIC ENGINE — Self-Test")
    print("=" * 70)
    
    for url, expected in test_urls:
        print(f"\nExpected: {expected}")
        print(explain_verdict(url))
        print("-" * 70)