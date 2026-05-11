"""
CP2 Feature Extractor — Phishing URL Detection
==============================================

Author: Cheah Qi Yang (22095483)
Project: Hybrid Machine Learning Framework for Phishing Detection
Module: src/feature_extractor.py

Purpose:
    Extracts 24 lexical features from a raw URL string for ML classification.
    Uses ONLY URL string analysis — no reputation lookup, no network calls.

Features grouped:
    Group 1 (Structural):    1-10  — Counts and lengths
    Group 2 (Statistical):   11-15 — Entropy, ratios
    Group 3 (Content):       16-20 — Keywords, security flags
    Group 4 (Super):         21-24 — Advanced detection (MITRE-mapped)

CP2 Critical Rule:
    These extractors operate on raw URLs only. The 54 pre-extracted columns
    in PhiUSIIL are NOT used — writing these from scratch IS the contribution.

Usage:
    from src.feature_extractor import extract_features
    features = extract_features("https://example.com/login")
    # Returns: dict of 24 feature_name -> numeric_value
"""

import math
import re
from collections import Counter
from typing import Dict
from urllib.parse import urlparse


# ============================================================
# CONSTANTS USED BY GROUP 3
# ============================================================

URL_SHORTENERS = {
    'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly', 'is.gd',
    'buff.ly', 'adf.ly', 'bit.do', 'mcaf.ee', 'short.link',
    'shorturl.at', 'rebrand.ly', 'tiny.cc', 'cutt.ly', 's.id',
    'rb.gy', 'soo.gd', 't.ly', 'lnkd.in'
}

LOGIN_KEYWORDS = {
    # Authentication-related
    'login', 'signin', 'sign-in', 'log-in', 'logon', 'auth',
    'authenticate', 'authentication',
    # Account-related
    'account', 'accounts', 'profile', 'user', 'customer',
    # Verification urgency
    'verify', 'verification', 'confirm', 'confirmation', 'validate',
    'validation', 'authorize',
    # Security theater
    'secure', 'security', 'safety', 'protected',
    # Financial/sensitive
    'banking', 'payment', 'wallet', 'billing', 'invoice',
    # Update/reset (password change lures)
    'update', 'reset', 'password', 'unlock',
    # Common targets
    'paypal', 'amazon', 'apple', 'microsoft', 'office365',
    'google', 'facebook', 'instagram', 'netflix'
}

SPOOFED_BRANDS = {
    'paypal', 'apple', 'amazon', 'microsoft', 'google',
    'facebook', 'instagram', 'netflix', 'ebay', 'linkedin',
    'twitter', 'whatsapp', 'wellsfargo', 'chase', 'bankofamerica',
    'citi', 'hsbc', 'maybank', 'cimb', 'publicbank',
    'office365', 'outlook', 'gmail', 'yahoo', 'dropbox',
    'adobe', 'github', 'spotify', 'steam'
}


# ============================================================
# GROUP 1: STRUCTURAL FEATURES (1-10)
# ============================================================

def url_length(url: str) -> int:
    """Feature #1: Total URL character count."""
    return len(url)


def qty_dot(url: str) -> int:
    """Feature #2: Count of '.' characters in URL."""
    return url.count('.')


def qty_hyphen(url: str) -> int:
    """Feature #3: Count of '-' characters in URL."""
    return url.count('-')


def qty_slash(url: str) -> int:
    """Feature #4: Count of '/' characters in URL."""
    return url.count('/')


def qty_at(url: str) -> int:
    """Feature #5: Count of '@' characters in URL."""
    return url.count('@')


def qty_question(url: str) -> int:
    """Feature #6: Count of '?' characters in URL."""
    return url.count('?')


def qty_equal(url: str) -> int:
    """Feature #7: Count of '=' characters in URL."""
    return url.count('=')


def qty_percent(url: str) -> int:
    """Feature #8: Count of '%' characters in URL."""
    return url.count('%')


def domain_length(url: str) -> int:
    """Feature #9: Length of registered domain (excluding subdomain/TLD)."""
    try:
        netloc = urlparse(url).netloc.split(':')[0]
        if netloc.startswith('www.'):
            netloc = netloc[4:]
        parts = netloc.split('.')
        if len(parts) >= 2:
            return len(parts[-2])
        return len(netloc)
    except Exception:
        return 0


def tld_length(url: str) -> int:
    """Feature #10: Length of TLD (top-level domain)."""
    try:
        netloc = urlparse(url).netloc.split(':')[0]
        parts = netloc.split('.')
        if len(parts) >= 2:
            return len(parts[-1])
        return 0
    except Exception:
        return 0


# ============================================================
# GROUP 2: STATISTICAL FEATURES (11-15)
# ============================================================

def _shannon_entropy(text: str) -> float:
    """Internal helper: Compute Shannon entropy of a string."""
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)
    return entropy


def url_entropy(url: str) -> float:
    """Feature #11: Shannon entropy of the full URL."""
    return round(_shannon_entropy(url), 4)


def domain_entropy(url: str) -> float:
    """Feature #12: Shannon entropy of just the domain portion."""
    try:
        netloc = urlparse(url).netloc.split(':')[0]
        if netloc.startswith('www.'):
            netloc = netloc[4:]
        return round(_shannon_entropy(netloc), 4)
    except Exception:
        return 0.0


def digit_ratio(url: str) -> float:
    """Feature #13: Ratio of digit characters to total URL length."""
    if not url:
        return 0.0
    digits = sum(c.isdigit() for c in url)
    return round(digits / len(url), 4)


def alpha_ratio(url: str) -> float:
    """Feature #14: Ratio of alphabetic characters to total URL length."""
    if not url:
        return 0.0
    alphas = sum(c.isalpha() for c in url)
    return round(alphas / len(url), 4)


def char_continuation_rate(url: str) -> float:
    """Feature #15: Rate of character-type transitions."""
    if len(url) < 2:
        return 0.0

    def _char_type(c: str) -> str:
        if c.isalpha():
            return 'alpha'
        if c.isdigit():
            return 'digit'
        return 'special'

    transitions = 0
    for i in range(1, len(url)):
        if _char_type(url[i]) != _char_type(url[i-1]):
            transitions += 1

    return round(transitions / len(url), 4)


# ============================================================
# GROUP 3: CONTENT FEATURES (16-20)
# ============================================================

def has_https(url: str) -> int:
    """Feature #16: Whether URL uses HTTPS (1) or HTTP (0)."""
    return 1 if url.lower().startswith('https://') else 0


def has_shortener(url: str) -> int:
    """Feature #17: Whether URL uses a known URL shortener service."""
    try:
        netloc = urlparse(url).netloc.lower().split(':')[0]
        if netloc.startswith('www.'):
            netloc = netloc[4:]
        return 1 if netloc in URL_SHORTENERS else 0
    except Exception:
        return 0


def has_login_keyword(url: str) -> int:
    """Feature #18: Whether URL contains login/verification keywords."""
    url_lower = url.lower()
    for keyword in LOGIN_KEYWORDS:
        if keyword in url_lower:
            return 1
    return 0


def is_ip(url: str) -> int:
    """Feature #19: Whether the URL uses an IP address instead of domain name."""
    try:
        netloc = urlparse(url).netloc.split(':')[0]
        if netloc.startswith('www.'):
            netloc = netloc[4:]

        # IPv4 pattern
        ipv4_pattern = r'^(?:\d{1,3}\.){3}\d{1,3}$'
        if re.match(ipv4_pattern, netloc):
            return 1

        # Hex-encoded IP pattern
        hex_ip_pattern = r'^(?:0x[0-9a-fA-F]+\.){3}0x[0-9a-fA-F]+$'
        if re.match(hex_ip_pattern, netloc):
            return 1

        # IPv6 detection
        if '[' in urlparse(url).netloc and ']' in urlparse(url).netloc:
            return 1

        return 0
    except Exception:
        return 0


def brand_in_path_not_domain(url: str) -> int:
    """Feature #20: Brand name in path but NOT in domain (spoofing)."""
    try:
        parsed = urlparse(url)
        netloc_lower = parsed.netloc.lower()
        path_text = (parsed.path + parsed.query + parsed.fragment).lower()

        for brand in SPOOFED_BRANDS:
            if brand in path_text and brand not in netloc_lower:
                return 1
        return 0
    except Exception:
        return 0


# ============================================================
# MASTER EXTRACTOR
# ============================================================

FEATURE_NAMES = [
    # Group 1: Structural (1-10)
    'url_length', 'qty_dot', 'qty_hyphen', 'qty_slash',
    'qty_at', 'qty_question', 'qty_equal', 'qty_percent',
    'domain_length', 'tld_length',
    # Group 2: Statistical (11-15)
    'url_entropy', 'domain_entropy', 'digit_ratio',
    'alpha_ratio', 'char_continuation_rate',
    # Group 3: Content (16-20)
    'has_https', 'has_shortener', 'has_login_keyword',
    'is_ip', 'brand_in_path_not_domain',
    # Group 4: Super features (21-24) — pending
]


def extract_features(url: str) -> Dict[str, float]:
    """
    Extract all features from a URL string.

    Args:
        url: Raw URL string (e.g., "https://example.com/login")

    Returns:
        Dictionary mapping feature_name -> numeric_value.

    Note:
        Currently implements features 1-20 (Groups 1-3).
        Group 4 (Super features 21-24) pending Days 4-6.
    """
    if not isinstance(url, str) or not url:
        return {feat: 0 for feat in FEATURE_NAMES}

    return {
        # Group 1: Structural
        'url_length': url_length(url),
        'qty_dot': qty_dot(url),
        'qty_hyphen': qty_hyphen(url),
        'qty_slash': qty_slash(url),
        'qty_at': qty_at(url),
        'qty_question': qty_question(url),
        'qty_equal': qty_equal(url),
        'qty_percent': qty_percent(url),
        'domain_length': domain_length(url),
        'tld_length': tld_length(url),
        # Group 2: Statistical
        'url_entropy': url_entropy(url),
        'domain_entropy': domain_entropy(url),
        'digit_ratio': digit_ratio(url),
        'alpha_ratio': alpha_ratio(url),
        'char_continuation_rate': char_continuation_rate(url),
        # Group 3: Content
        'has_https': has_https(url),
        'has_shortener': has_shortener(url),
        'has_login_keyword': has_login_keyword(url),
        'is_ip': is_ip(url),
        'brand_in_path_not_domain': brand_in_path_not_domain(url),
    }


# ============================================================
# QUICK SELF-TEST
# ============================================================

if __name__ == "__main__":
    test_urls = [
        ("https://www.google.com", "legitimate, simple"),
        ("https://www.paypal-secure-login.evil.tk/account/verify?id=12345", "phishing"),
        ("http://192.168.1.1/admin", "IP-based suspicious"),
        ("https://docs.python.org/3/library/urllib.parse.html", "legitimate, long path"),
        ("https://bit.ly/3xK9pQz", "URL shortener (should flag)"),
        ("https://evil.com/paypal/login/verify", "brand spoofing in path"),
    ]

    print("=" * 70)
    print("FEATURE EXTRACTOR — Self-Test")
    print("=" * 70)

    for url, label in test_urls:
        print(f"\nURL: {url[:80]}")
        print(f"Type: {label}")
        features = extract_features(url)
        for name, value in features.items():
            print(f"  {name:25s}: {value}")