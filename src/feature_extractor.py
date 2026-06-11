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
from urllib.parse import urlparse, unquote
import Levenshtein


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
    'google', 'facebook', 'instagram', 'netflix',
    # Malay-language phishing lures (local contribution)
    'selamat', 'akaun', 'perbankan', 'pengesahan', 'kemaskini',
    'kata', 'laluan', 'masuk', 'semak', 'sahkan',
    'tuntutan', 'hadiah', 'wang', 'bayaran', 'kredit'
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
# GROUP 4: SUPER FEATURES (21-24) — Advanced detection
# ============================================================
# Complex features mapped to MITRE ATT&CK techniques.
# These are the CP2 "innovation" — beyond standard ML features.


def recursive_decode_depth(url: str, max_iterations: int = 10) -> int:
    """
    Feature #21: How many URL-decode iterations until the string stabilizes.
    
    Predictive signal: STRONG (when > 1)
    MITRE ATT&CK: T1027.001 (Obfuscated Files or Information: Binary Padding)
    
    Detects multi-layer URL encoding used to evade pattern-matching filters.
    Phishing example: a URL might be encoded 2-3 times to hide payload.
    
    Algorithm:
        1. Decode the URL once.
        2. Compare with original.
        3. If different, repeat.
        4. If same, return iteration count.
    
    Interpretation:
        0  -> No encoding present (plain URL)
        1  -> Standard single-level encoding (normal, common)
        2+ -> Multi-layer obfuscation (suspicious)
        3+ -> Very suspicious (intentional evasion)
    
    Safety:
        - Capped at max_iterations to prevent infinite loops
        - Handles malformed encoding gracefully
    """
    if not url:
        return 0
    
    current = url
    depth = 0
    
    try:
        for _ in range(max_iterations):
            decoded = unquote(current)
            if decoded == current:
                # No more decoding possible — stable
                break
            current = decoded
            depth += 1
        return depth
    except Exception:
        return 0


# New advanced helpers (IDN / typosquat)

# Top brands for typosquat detection (extend as needed)
TOP_BRANDS = [
    'paypal', 'apple', 'amazon', 'microsoft', 'google',
    'facebook', 'instagram', 'netflix', 'ebay', 'linkedin',
    'twitter', 'whatsapp', 'youtube', 'wikipedia', 'reddit',
    'wellsfargo', 'chase', 'bankofamerica', 'citi', 'hsbc',
    'maybank', 'cimb', 'publicbank', 'rhb', 'ambank',
    'office365', 'outlook', 'gmail', 'yahoo', 'dropbox',
    'adobe', 'github', 'spotify', 'steam', 'discord',
    'tiktok', 'snapchat', 'pinterest', 'tumblr', 'twitch',
    'paypal', 'venmo', 'cashapp', 'zelle', 'stripe',
    'shopify', 'walmart', 'target', 'bestbuy', 'costco',
    'fedex', 'ups', 'usps', 'dhl', 'amazon',
    'icloud', 'onedrive', 'googledrive', 'mega', 'box',
    'zoom', 'teams', 'slack', 'webex', 'meet',
    'lazada', 'shopee', 'grab', 'foodpanda', 'tngdigital',
    'maxis', 'celcom', 'digi', 'unifi', 'astro',
    'sunway', 'taylors', 'monash', 'inti', 'help',
    'binance', 'coinbase', 'kraken', 'bybit', 'okx'
]


def idn_homograph_flag(url: str) -> int:
    """
    Feature #22: Punycode/IDN homograph attack detection.
    MITRE ATT&CK: T1036.007 (Masquerading: Double File Extension)
    
    Returns 1 if any domain part contains 'xn--' (Punycode prefix).
    Punycode encodes Unicode chars that look like Latin but aren't.
    """
    try:
        netloc = urlparse(url).netloc.lower().split(':')[0]
        if 'xn--' in netloc:
            return 1
        return 0
    except Exception:
        return 0


def levenshtein_min(url: str) -> int:
    """
    Feature #23: Minimum edit distance from domain to known brands.
    MITRE ATT&CK: T1566.002 (Spearphishing Link: Typosquatting)
    
    Returns smallest Levenshtein distance between the registered domain
    and any brand in TOP_BRANDS list.
    """
    try:
        netloc = urlparse(url).netloc.lower().split(':')[0]
        if netloc.startswith('www.'):
            netloc = netloc[4:]
        # Get registered domain part (before TLD)
        parts = netloc.split('.')
        if len(parts) >= 2:
            domain = parts[-2]
        else:
            domain = netloc
        
        if not domain:
            return 99  # sentinel for invalid
        
        # Compute min distance across all brands
        min_dist = min(Levenshtein.distance(domain, brand) for brand in TOP_BRANDS)
        return min_dist
    except Exception:
        return 99

# TLD risk scores: 1.0 = high phishing concentration, 0.0 = legitimate
# Built from Week 1 PhiUSIIL exploration + Spamhaus DBL reference
TLD_RISK_SCORES = {
    # 100% phishing in PhiUSIIL (max risk)
    'cf': 1.0, 'ml': 1.0, 'gq': 1.0, 'cfd': 1.0,
    'icu': 1.0, 'page': 1.0,
    # 95-99% phishing
    'top': 0.99, 'site': 0.99, 'ga': 0.99, 'link': 0.99,
    'gd': 0.99, 'fun': 0.98, 'dev': 0.98, 'xyz': 0.98,
    'tk': 0.98, 'ly': 0.98, 'app': 0.97, 'cloud': 0.97,
    'my.id': 0.97,
    # Mid-risk (50-90%)
    'ru': 0.77, 'su': 0.75, 'pw': 0.70, 'work': 0.65,
    'click': 0.65, 'download': 0.65, 'review': 0.60,
    'club': 0.88, 'cn': 0.85,
    # Mixed (40-60%)
    'com': 0.40, 'net': 0.43, 'co': 0.45, 'io': 0.91,
    # Low risk (<20%)
    'org': 0.12, 'info': 0.20, 'biz': 0.25,
    'me': 0.79, 'tv': 0.15,
    # Country-code TLDs (mostly legitimate)
    'uk': 0.05, 'de': 0.05, 'fr': 0.05, 'it': 0.05,
    'nl': 0.05, 'es': 0.05, 'ca': 0.05, 'au': 0.05,
    'jp': 0.05, 'kr': 0.05, 'br': 0.05, 'mx': 0.05,
    'my': 0.05, 'sg': 0.05, 'th': 0.05, 'id': 0.05,
    # Trusted (essentially never phishing)
    'edu': 0.0, 'gov': 0.0, 'mil': 0.0,
    'co.uk': 0.0, 'com.au': 0.0, 'com.my': 0.0,
    'org.uk': 0.0, 'com.br': 0.0, 'co.jp': 0.0,
}

DEFAULT_TLD_RISK = 0.5  # unknown TLD = neutral


def tld_risk_score(url: str) -> float:
    """
    Feature #24: Risk score for URL's TLD based on phishing concentration.
    MITRE ATT&CK: T1583.001 (Acquire Infrastructure: Domains)
    
    Maps TLD to risk score (0.0-1.0):
        1.0 = TLD with 100% phishing in training data (.cf, .ml, .gq)
        0.5 = unknown TLD (neutral default)
        0.0 = trusted TLD (.edu, .gov, .gov.uk)
    """
    try:
        netloc = urlparse(url).netloc.lower().split(':')[0]
        parts = netloc.split('.')
        if len(parts) < 2:
            return DEFAULT_TLD_RISK
        
        # Try compound TLD first (co.uk, com.my)
        if len(parts) >= 3:
            compound_tld = '.'.join(parts[-2:])
            if compound_tld in TLD_RISK_SCORES:
                return TLD_RISK_SCORES[compound_tld]
        
        # Fallback to single TLD
        tld = parts[-1]
        return TLD_RISK_SCORES.get(tld, DEFAULT_TLD_RISK)
    except Exception:
        return DEFAULT_TLD_RISK

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
    # Group 4: Super (21-24)
    'recursive_decode_depth',
    'idn_homograph_flag',
    'levenshtein_min',
    'tld_risk_score',
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
        # Group 4: Super
        'recursive_decode_depth': recursive_decode_depth(url),
        'idn_homograph_flag': idn_homograph_flag(url),
        'levenshtein_min': levenshtein_min(url),
        'tld_risk_score': tld_risk_score(url),
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
        ("https://evil.com/login", "plain URL (depth=0)"),
        ("https://evil.com%2Flogin", "single-encoded (depth=1)"),
        ("https://evil.com%252Flogin", "double-encoded (depth=2, suspicious)"),
        ("https://evil.com%25252Flogin", "triple-encoded (depth=3, very suspicious)"),
        ("https://xn--pypal-4ve.com/login", "IDN homograph (Cyrillic spoof)"),
        ("https://paypa1.com/login", "typosquat distance=1"),
        ("https://g00gle.com", "typosquat distance=2"),
        ("https://paypal.com", "legitimate brand (distance=0)"),
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