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

from urllib.parse import urlparse
from typing import Dict


# ============================================================
# GROUP 1: STRUCTURAL FEATURES (1-10)
# ============================================================
# Pure character/length counts. Fast, deterministic, foundational.
# Predictive signals from Week 1 exploration:
#   - Phishing URLs avg 46 chars vs 27 for legitimate (70% longer)
#   - Phishing URLs have higher punctuation density


def url_length(url: str) -> int:
    """
    Feature #1: Total URL character count.
    
    Predictive signal: STRONG
    Week 1 finding: phishing mean=46 chars, legitimate mean=27 chars.
    Max phishing URL was 6097 chars (encoded payload).
    """
    return len(url)


def qty_dot(url: str) -> int:
    """
    Feature #2: Count of '.' characters in URL.
    
    Predictive signal: MEDIUM
    Phishing often has more dots (subdomain nesting, IP addresses).
    Example: secure.login.paypal.evil.com has 4 dots.
    """
    return url.count('.')


def qty_hyphen(url: str) -> int:
    """
    Feature #3: Count of '-' characters in URL.
    
    Predictive signal: MEDIUM
    Phishing uses hyphens to construct fake brand names.
    Example: paypal-secure-login.com vs paypal.com
    """
    return url.count('-')


def qty_slash(url: str) -> int:
    """
    Feature #4: Count of '/' characters in URL.
    
    Predictive signal: MEDIUM
    Phishing uses deep paths to hide payload location.
    Example: /wp-content/jam/ichiemagiksouthwest123.html
    """
    return url.count('/')


def qty_at(url: str) -> int:
    """
    Feature #5: Count of '@' characters in URL.
    
    Predictive signal: STRONG (when present)
    '@' in URL is rare but highly suspicious. Used for:
      - Embedding email in URL path (credential harvest)
      - URL spoofing: http://legitimate.com@evil.com
    Week 1 finding example: leszek.arekhasnik.pl/add/email@example.com
    """
    return url.count('@')


def qty_question(url: str) -> int:
    """
    Feature #6: Count of '?' characters in URL.
    
    Predictive signal: MEDIUM
    Multiple '?' is invalid HTTP. Phishing often has malformed query strings.
    Note: most URLs have at most 1 (the query separator).
    """
    return url.count('?')


def qty_equal(url: str) -> int:
    """
    Feature #7: Count of '=' characters in URL.
    
    Predictive signal: MEDIUM
    Many '=' indicates many query parameters (tracking, callback URLs,
    embedded encoded data).
    """
    return url.count('=')


def qty_percent(url: str) -> int:
    """
    Feature #8: Count of '%' characters in URL.
    
    Predictive signal: STRONG
    '%' is URL encoding (%20=space, %2F=slash). High counts indicate:
      - Encoded payloads to bypass filters
      - Obfuscated paths
    Example: /wx%20qz%20xx%20ww%20qxz%20wes%20wsed/ (from Week 1 outliers)
    """
    return url.count('%')


def domain_length(url: str) -> int:
    """
    Feature #9: Length of registered domain (excluding subdomain and TLD).
    
    Predictive signal: WEAK (counter-intuitive)
    Week 1 finding: legitimate domains AVG 10.83 chars vs phishing 8.98.
    Legitimate has longer domain names (real businesses).
    Phishing uses short throwaway domains.
    Kept for completeness; XGBoost will weight it appropriately.
    """
    try:
        netloc = urlparse(url).netloc
        # Remove port if present (e.g., example.com:8080 -> example.com)
        netloc = netloc.split(':')[0]
        # Remove www. prefix
        if netloc.startswith('www.'):
            netloc = netloc[4:]
        # Get the domain part (before last dot for TLD)
        parts = netloc.split('.')
        if len(parts) >= 2:
            return len(parts[-2])  # second-to-last is the registered domain
        return len(netloc)
    except Exception:
        return 0


def tld_length(url: str) -> int:
    """
    Feature #10: Length of TLD (top-level domain).
    
    Predictive signal: WEAK
    Common phishing TLDs: .tk (2), .ml (2), .gq (2), .cf (2) — short.
    Common legitimate TLDs: .com (3), .org (3), .edu (3), .gov (3).
    Long TLDs: .info (4), .online (6), .website (7).
    Marginal predictive power but cheap to compute.
    """
    try:
        netloc = urlparse(url).netloc.split(':')[0]
        parts = netloc.split('.')
        if len(parts) >= 2:
            return len(parts[-1])  # last part is TLD
        return 0
    except Exception:
        return 0

# ============================================================
# GROUP 2: STATISTICAL FEATURES (11-15)
# ============================================================
# Mathematical properties of URL strings.
# Predictive signals from Week 1 exploration:
#   - digit_ratio: 31x higher in phishing (0.0636 vs 0.0020)
#   - alpha_ratio: 5% lower in phishing (0.7472 vs 0.8000)
#   - Entropy: phishing tends higher (random-looking strings)


import math
from collections import Counter


def _shannon_entropy(text: str) -> float:
    """
    Internal helper: Compute Shannon entropy of a string.
    
    Shannon entropy measures information density / randomness.
    Pure repetition (aaaa) = 0 bits.
    Random uniform chars (abcdef) = ~2.58 bits.
    High randomness indicates encoded payloads or random domain generation.
    """
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
    """
    Feature #11: Shannon entropy of the full URL.
    
    Predictive signal: STRONG
    MITRE ATT&CK: T1027 (Obfuscated Files or Information)
    
    Higher entropy indicates random-looking URLs (base64 payloads,
    random subdomain generation, encoded data). Typical ranges:
      - Legitimate URLs: 3.5 - 4.5 bits
      - Phishing with encoded payload: 4.5 - 5.5+ bits
      - Pure DNS (rare): 4.0 bits
    """
    return round(_shannon_entropy(url), 4)


def domain_entropy(url: str) -> float:
    """
    Feature #12: Shannon entropy of just the domain portion.
    
    Predictive signal: MEDIUM
    
    Catches domain generation algorithms (DGAs) and random hex domains.
    Examples:
      - google.com -> low entropy (real word)
      - v2cde1b0d66c767a23cfb34c14552836d3.ws -> high entropy (random hex)
    """
    try:
        netloc = urlparse(url).netloc.split(':')[0]
        # Strip www. prefix
        if netloc.startswith('www.'):
            netloc = netloc[4:]
        return round(_shannon_entropy(netloc), 4)
    except Exception:
        return 0.0


def digit_ratio(url: str) -> float:
    """
    Feature #13: Ratio of digit characters to total URL length.
    
    Predictive signal: VERY STRONG (top feature)
    Week 1 finding: 31x higher in phishing (6.36% vs 0.20%).
    
    Catches IP addresses, hex strings, random tokens.
    Range: 0.0 to 1.0.
    """
    if not url:
        return 0.0
    digits = sum(c.isdigit() for c in url)
    return round(digits / len(url), 4)


def alpha_ratio(url: str) -> float:
    """
    Feature #14: Ratio of alphabetic characters to total URL length.
    
    Predictive signal: WEAK
    Week 1 finding: only 5% gap (74.72% phishing vs 80.00% legitimate).
    
    Lower alpha ratio = more noise/special chars in URL.
    Kept for completeness — XGBoost will weight it appropriately.
    Range: 0.0 to 1.0.
    """
    if not url:
        return 0.0
    alphas = sum(c.isalpha() for c in url)
    return round(alphas / len(url), 4)


def char_continuation_rate(url: str) -> float:
    """
    Feature #15: Rate of consecutive same-character-type sequences.
    
    Predictive signal: MEDIUM
    MITRE ATT&CK: T1027 (Obfuscated Files or Information)
    
    Measures how often the URL transitions between character types
    (letter -> digit, digit -> special, etc.). Lower rate = smoother
    URLs (legitimate). Higher rate = chaotic transitions (phishing).
    
    Example logic:
      "google" (all letters)         -> 0 transitions / 6 chars = 0.0
      "ab1cd2ef3" (constant flip)    -> 8 transitions / 9 chars = 0.89
    """
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
# MASTER EXTRACTOR (will grow as more groups are added)
# ============================================================

def extract_features(url: str) -> Dict[str, float]:
    """
    Extract all 24 features from a URL string.
    
    Args:
        url: Raw URL string (e.g., "https://example.com/login")
    
    Returns:
        Dictionary mapping feature_name -> numeric_value (24 keys total)
    
    Note:
        Currently implements features 1-10 (Group 1: Structural).
        Will expand as Groups 2-4 are added in subsequent days.
    """
    if not isinstance(url, str) or not url:
        # Return zero-vector for invalid input (defensive)
        return {feat: 0 for feat in [
            'url_length', 'qty_dot', 'qty_hyphen', 'qty_slash',
            'qty_at', 'qty_question', 'qty_equal', 'qty_percent',
            'domain_length', 'tld_length',
            'url_entropy', 'domain_entropy', 'digit_ratio',
            'alpha_ratio', 'char_continuation_rate'
        ]}
    
    return {
        # Group 1: Structural (1-10)
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
        # Group 2: Statistical (11-15)
        'url_entropy': url_entropy(url),
        'domain_entropy': domain_entropy(url),
        'digit_ratio': digit_ratio(url),
        'alpha_ratio': alpha_ratio(url),
        'char_continuation_rate': char_continuation_rate(url),
        # Groups 3-4 will be added here in subsequent days
    }
        

# ============================================================
# QUICK SELF-TEST (run with: python src/feature_extractor.py)
# ============================================================

if __name__ == "__main__":
    # Sanity check — run this file directly to verify features work
    test_urls = [
        ("https://www.google.com", "legitimate, simple"),
        ("https://www.paypal-secure-login.evil.tk/account/verify?id=12345", "phishing"),
        ("http://192.168.1.1/admin", "IP-based suspicious"),
        ("https://docs.python.org/3/library/urllib.parse.html", "legitimate, long path"),
    ]
    
    print("=" * 70)
    print("FEATURE EXTRACTOR — Self-Test")
    print("=" * 70)
    
    for url, label in test_urls:
        print(f"\nURL: {url[:80]}")
        print(f"Type: {label}")
        features = extract_features(url)
        for name, value in features.items():
            print(f"  {name:20s}: {value}")