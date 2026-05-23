"""
PhishLens — Hybrid Phishing Detection Dashboard (Full Rewrite)
================================================================
Author: Cheah Qi Yang (22095483)
Sunway University CP2 Capstone

Features:
    1. Specific quick test names with attack type labels
    2. Plain English + jargon explanation toggle
    3. MITRE inferred from features (even when Layer 1 clean)
    4. Feature group breakdown panel
    5. Useful threat landscape sidebar
    6. Batch URL upload with downloadable results
    7. Risk score gauge visual
    8. Plain English verdict summary
    9. Top 3 suspicious features highlighted
    10. Scan history (last 5 URLs)

Run: streamlit run dashboard/app.py
"""

import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from feature_extractor import extract_features, FEATURE_NAMES
from hybrid_scorer import hybrid_predict, _is_trusted_domain
from heuristic_engine import RULES

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="PhishLens",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu, footer { visibility: hidden; }
.block-container { padding: 1.5rem 2.5rem; max-width: 1300px; }

/* Verdict banners */
.verdict-box {
    padding: 18px 28px; border-radius: 12px;
    font-size: 24px; font-weight: 800;
    text-align: center; letter-spacing: 2px;
    margin: 12px 0; text-transform: uppercase;
}
.v-critical { background: linear-gradient(135deg,#7f0000,#ff1744); color:#fff; border-left: 6px solid #ff6b6b; }
.v-high     { background: linear-gradient(135deg,#bf360c,#ff6d00); color:#fff; border-left: 6px solid #ffab40; }
.v-medium   { background: linear-gradient(135deg,#e65100,#ffca28); color:#111; border-left: 6px solid #ffd740; }
.v-low      { background: linear-gradient(135deg,#1b5e20,#66bb6a); color:#fff; border-left: 6px solid #69f0ae; }
.v-safe     { background: linear-gradient(135deg,#004d40,#00bfa5); color:#fff; border-left: 6px solid #64ffda; }

/* Plain English summary box */
.plain-box {
    background: #1a1a2e; border: 1px solid #444;
    border-left: 4px solid #4fc3f7;
    border-radius: 8px; padding: 14px 18px; margin: 8px 0;
    font-size: 15px; line-height: 1.6;
}
.jargon-box {
    background: #0d1117; border: 1px solid #333;
    border-left: 4px solid #7c4dff;
    border-radius: 8px; padding: 12px 18px; margin: 8px 0;
    font-size: 13px; font-family: 'Courier New', monospace;
    color: #aaa;
}

/* MITRE badge */
.badge-mitre {
    display:inline-block; background:#1a237e; color:#fff;
    padding:4px 14px; border-radius:20px;
    font-size:12px; font-weight:700; margin:3px;
    border: 1px solid #3949ab;
}
.badge-mitre-desc {
    display:inline-block; background:#0d1b2a; color:#90caf9;
    padding:3px 10px; border-radius:6px;
    font-size:11px; margin:2px 4px;
}

/* Rule badge */
.badge-rule {
    display:inline-block; background:#4a0000; color:#ff8a80;
    padding:4px 14px; border-radius:20px;
    font-size:12px; font-weight:700; margin:3px;
    border: 1px solid #b71c1c;
}

/* Feature highlight card */
.feat-card {
    background: #1e1e2e; border: 1px solid #333;
    border-radius: 8px; padding: 12px 16px; margin: 6px 0;
    border-left: 3px solid #ff5252;
}
.feat-card-safe {
    background: #1e2e1e; border: 1px solid #333;
    border-radius: 8px; padding: 12px 16px; margin: 6px 0;
    border-left: 3px solid #00e676;
}

/* Gauge bar */
.gauge-container { margin: 10px 0; }
.gauge-label { font-size: 12px; color: #aaa; margin-bottom: 4px; }
.gauge-bar {
    height: 12px; border-radius: 6px;
    background: linear-gradient(90deg, #00e676 0%, #ffca28 50%, #ff1744 100%);
    position: relative; overflow: visible;
}
.gauge-marker {
    position: absolute; top: -4px;
    width: 4px; height: 20px;
    background: white; border-radius: 2px;
    box-shadow: 0 0 6px rgba(255,255,255,0.8);
}

/* History item */
.history-item {
    background: #1a1a2e; border: 1px solid #333;
    border-radius: 6px; padding: 8px 12px; margin: 4px 0;
    font-size: 12px; cursor: pointer;
}

/* Section title */
.section-title {
    font-size: 16px; font-weight: 700;
    color: #90caf9; margin: 16px 0 8px 0;
    border-bottom: 1px solid #333; padding-bottom: 6px;
}

/* Batch results */
.batch-safe { color: #00e676; }
.batch-critical { color: #ff5252; }
.batch-high { color: #ff9800; }
.batch-medium { color: #ffeb3b; }
.batch-low { color: #69f0ae; }
</style>
""", unsafe_allow_html=True)

# ── MITRE technique descriptions ───────────────────────────
MITRE_INFO = {
    'T1566.002': {
        'name': 'Spearphishing Link',
        'plain': 'Attacker sent a deceptive link disguised as a trusted website to steal your credentials',
        'jargon': 'T1566.002 — Spearphishing via Link: adversary crafts malicious URL mimicking legitimate service'
    },
    'T1036.007': {
        'name': 'Masquerading (Homograph)',
        'plain': 'URL uses foreign characters that look identical to English letters — your browser shows paypal.com but it is a fake',
        'jargon': 'T1036.007 — Masquerading: IDN homograph attack using Unicode lookalike characters (xn-- Punycode encoding)'
    },
    'T1027.001': {
        'name': 'Obfuscated URL Encoding',
        'plain': 'The URL has been scrambled multiple times to hide its true destination from security filters',
        'jargon': 'T1027.001 — Obfuscated Files: multi-layer percent-encoding applied to evade pattern-matching detection'
    },
    'T1557': {
        'name': 'Adversary-in-the-Middle (AiTM)',
        'plain': 'This URL matches known phishing kit patterns that intercept your login session — attacker captures your credentials AND your 2FA code',
        'jargon': 'T1557 — Adversary-in-the-Middle: AiTM phishing proxy kit (Tycoon/Starkiller/Mamba pattern) relays credentials in real-time'
    },
    'T1583.001': {
        'name': 'Suspicious Domain Infrastructure',
        'plain': 'The website ending (.tk, .ml, .cf etc.) is used almost exclusively by attackers because it is free and anonymous',
        'jargon': 'T1583.001 — Acquire Infrastructure: high-risk TLD with documented phishing concentration (Spamhaus DBL + APWG cross-referenced)'
    },
    'T1071.001': {
        'name': 'IP Address Instead of Domain',
        'plain': 'Legitimate websites always use a name like google.com — using a raw IP address (like 192.168.1.1) is a classic attack sign',
        'jargon': 'T1071.001 — Application Layer Protocol: URL uses raw IPv4/IPv6 address as host, bypassing domain reputation checks'
    },
}

# ── Feature explanations (plain + jargon) ─────────────────
FEATURE_EXPLANATIONS = {
    'tld_risk_score': {
        'plain': 'Website ending (.tk, .ml, .cf) is used almost exclusively by phishing sites',
        'jargon': 'TLD risk score {val:.2f}/1.0 — derived from PhiUSIIL frequency + Spamhaus DBL cross-reference',
        'mitre': 'T1583.001'
    },
    'digit_ratio': {
        'plain': 'URL contains an unusually high number of digits — phishing URLs average 31x more digits than legitimate ones',
        'jargon': 'Digit ratio {val:.4f} — 31x gap between phishing (0.0636) and legitimate (0.0020) class means',
        'mitre': 'T1566.002'
    },
    'levenshtein_min': {
        'plain': 'Domain name is only {val:.0f} character(s) away from a real brand — classic typosquatting trick',
        'jargon': 'Levenshtein distance {val:.0f} — minimum edit distance to known brand list (e.g. paypa1.com → paypal)',
        'mitre': 'T1566.002'
    },
    'url_length': {
        'plain': 'URL is unusually long ({val:.0f} characters) — legitimate homepages average 27 characters',
        'jargon': 'URL length {val:.0f} chars — phishing mean 46.24 vs legitimate mean 27.23 (PhiUSIIL baseline)',
        'mitre': 'T1566.002'
    },
    'url_entropy': {
        'plain': 'URL looks randomly generated — high randomness is a sign of automated phishing kit domain generation',
        'jargon': 'Shannon entropy {val:.4f} — high entropy indicates DGA (Domain Generation Algorithm) origin',
        'mitre': 'T1027.001'
    },
    'recursive_decode_depth': {
        'plain': 'URL is hidden inside {val:.0f} layers of encoding — attacker scrambled it to bypass security filters',
        'jargon': 'Recursive decode depth {val:.0f} — multi-layer percent-encoding detected (MITRE T1027.001 obfuscation)',
        'mitre': 'T1027.001'
    },
    'idn_homograph_flag': {
        'plain': 'URL uses Punycode encoding — browser shows a familiar brand name but it points to a fake site',
        'jargon': 'IDN homograph detected (xn-- prefix) — Unicode lookalike substitution (e.g. Cyrillic а vs Latin a)',
        'mitre': 'T1036.007'
    },
    'is_ip': {
        'plain': 'URL uses a raw IP address instead of a domain name — legitimate sites never do this for user-facing pages',
        'jargon': 'IPv4/IPv6 address as host — bypasses domain reputation systems, MITRE T1071.001',
        'mitre': 'T1071.001'
    },
    'has_login_keyword': {
        'plain': 'URL contains words like login, verify, or account — phishing sites use these to trick users into entering credentials',
        'jargon': 'Login keyword detected in URL path — social engineering lure pattern (T1566.002)',
        'mitre': 'T1566.002'
    },
    'brand_in_path_not_domain': {
        'plain': 'A real brand name appears in the page path but the actual domain is different — classic fake login page',
        'jargon': 'Brand name in URL path but not in registered domain — spoof indicator (T1566.002)',
        'mitre': 'T1566.002'
    },
    'qty_at': {
        'plain': 'URL contains an @ symbol — attackers use this trick to hide the real destination after the @ sign',
        'jargon': 'Credential embedding via @ symbol — browsers resolve URL after @ as actual host',
        'mitre': 'T1566.002'
    },
}

# ── MITRE inference from features ─────────────────────────
def infer_mitre_from_features(features: dict, verdict: str) -> list:
    """Infer MITRE techniques from feature values when Layer 1 did not trigger."""
    techniques = []

    if features.get('idn_homograph_flag', 0) == 1:
        techniques.append('T1036.007')
    if features.get('recursive_decode_depth', 0) >= 2:
        techniques.append('T1027.001')
    if features.get('is_ip', 0) == 1:
        techniques.append('T1071.001')
    if features.get('tld_risk_score', 0) >= 0.9:
        techniques.append('T1583.001')
    if features.get('levenshtein_min', 99) <= 2:
        techniques.append('T1566.002')
    if features.get('brand_in_path_not_domain', 0) == 1:
        techniques.append('T1566.002')
    if features.get('has_login_keyword', 0) == 1:
        techniques.append('T1566.002')
    if features.get('qty_at', 0) >= 1:
        techniques.append('T1566.002')

    # If phishing verdict but no specific technique found, default to spearphishing
    if verdict in ['CRITICAL', 'HIGH', 'MEDIUM'] and not techniques:
        techniques.append('T1566.002')

    # Deduplicate preserving order
    seen = set()
    return [t for t in techniques if not (t in seen or seen.add(t))]


# ── Top suspicious features ────────────────────────────────
def get_top_suspicious_features(features: dict, verdict: str) -> list:
    """Return top 3 most suspicious feature explanations."""
    suspicious = []

    checks = [
        ('idn_homograph_flag',    lambda v: v == 1),
        ('is_ip',                 lambda v: v == 1),
        ('recursive_decode_depth',lambda v: v >= 2),
        ('tld_risk_score',        lambda v: v >= 0.7),
        ('levenshtein_min',       lambda v: v <= 3),
        ('brand_in_path_not_domain', lambda v: v == 1),
        ('digit_ratio',           lambda v: v > 0.05),
        ('has_login_keyword',     lambda v: v == 1),
        ('url_entropy',           lambda v: v > 4.5),
        ('url_length',            lambda v: v > 50),
        ('qty_at',                lambda v: v >= 1),
    ]

    for feat_name, condition in checks:
        val = features.get(feat_name, 0)
        if condition(val) and feat_name in FEATURE_EXPLANATIONS:
            info = FEATURE_EXPLANATIONS[feat_name]
            suspicious.append({
                'feature': feat_name,
                'value': val,
                'plain': info['plain'].replace('{val:.0f}', str(int(val)))
                                      .replace('{val:.2f}', f'{val:.2f}')
                                      .replace('{val:.4f}', f'{val:.4f}'),
                'jargon': info['jargon'].replace('{val:.0f}', str(int(val)))
                                        .replace('{val:.2f}', f'{val:.2f}')
                                        .replace('{val:.4f}', f'{val:.4f}'),
                'mitre': info.get('mitre', '')
            })
        if len(suspicious) >= 3:
            break

    return suspicious


# ── Risk gauge ─────────────────────────────────────────────
def render_gauge(score: float, cvss: float):
    pct = score * 100
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"""
        <div class="gauge-container">
            <div class="gauge-label">Risk Level (0 = Safe → 10 = Critical)</div>
            <div style="position:relative; height:20px;">
                <div style="height:12px; border-radius:6px; margin-top:4px;
                    background: linear-gradient(90deg, #00e676 0%, #ffca28 50%, #ff1744 100%);">
                </div>
                <div style="position:absolute; top:0px; left:{min(pct, 97)}%;
                    width:6px; height:20px; background:white; border-radius:3px;
                    box-shadow: 0 0 8px rgba(255,255,255,0.9);">
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.metric("Phishing Score", f"{score:.3f}", help="0.0 = definitely safe, 1.0 = definitely phishing")
    with col3:
        st.metric("CVSS Severity", f"{cvss} / 10.0", help="Industry-standard severity scale used in cybersecurity")


# ── Model loader ───────────────────────────────────────────
@st.cache_resource(show_spinner="Loading PhishLens models...")
def load_models():
    xgb = joblib.load('models/xgboost_model.joblib')
    rf  = joblib.load('models/rf_model.joblib')
    return xgb, rf

@st.cache_resource(show_spinner="Initializing SHAP explainer...")
def load_explainer(_xgb_model):
    return shap.TreeExplainer(_xgb_model)


# ── SHAP plot ──────────────────────────────────────────────
def shap_waterfall(features, xgb_model, explainer):
    X = pd.DataFrame([features])[FEATURE_NAMES]
    sv = explainer(X)
    plt.style.use('dark_background')
    fig, _ = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('#0e1117')
    shap.plots.waterfall(sv[0], max_display=12, show=False)
    plt.title("Which features caused this verdict?", fontsize=12, pad=10, color='white')
    plt.tight_layout()
    return fig


# ── Sidebar ────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown("## PhishLens")
        st.markdown("*Hybrid Phishing Detection System*")
        st.markdown("---")
        st.markdown("**Project**")
        st.markdown("Cheah Qi Yang (22095483)")
        st.markdown("Sunway University | CP2 Capstone")
        st.markdown("Supervisor: Dr Mohd Firdaus Roslan")
        st.markdown("---")

        st.markdown("**How It Works**")
        st.markdown("""
**Layer 1 — Rule Engine**
Instantly flags URLs matching known attack patterns. No AI needed for obvious threats.

**Layer 2 — AI Detection**
XGBoost + Random Forest ensemble trained on 188,296 URLs. Catches subtle phishing that rules miss.

**Layer 3 — Smart Combiner**
If rules say suspicious but AI says safe, the system escalates — AI alone can be fooled.
        """)
        st.markdown("---")

        st.markdown("**Severity Guide**")
        st.markdown("""
| Score | Level | Meaning |
|-------|-------|---------|
| ≥0.90 | 🔴 CRITICAL | Active phishing attack |
| ≥0.70 | 🟠 HIGH | Very likely phishing |
| ≥0.40 | 🟡 MEDIUM | Suspicious, proceed carefully |
| ≥0.20 | 🟢 LOW | Minor signals only |
| <0.20 | ✅ SAFE | No threats detected |
        """)
        st.markdown("---")

        st.markdown("**2026 Active Threats**")
        st.markdown("""
Your system detects these current attack types:

🔴 **Starkiller AiTM Kit**
Successor to Tycoon 2FA (taken down Jan 2026). Intercepts login AND 2FA codes simultaneously.

🔴 **Mamba 2FA**
Phishing-as-a-service kit. Targets Microsoft 365 and Google accounts.

🟠 **HTTPS Laundering**
Attackers get free SSL certificates to show the padlock icon — HTTPS no longer means safe.

🟠 **IDN Homograph Attacks**
Fake domains using Cyrillic/Greek characters that look identical to English in browser address bar.
        """)
        st.markdown("---")
        st.markdown("**Performance**")
        st.markdown("""
- F1 Score: **0.9977**
- ROC-AUC: **0.9988**
- Precision: **99.97%**
- OpenPhish Detection: **100%** (300 URLs)
- Adversarial Retention: **100%** (5 mutations)
        """)


# ── Scan history ───────────────────────────────────────────
def init_history():
    if 'scan_history' not in st.session_state:
        st.session_state.scan_history = []

def add_to_history(url, verdict, score):
    history = st.session_state.scan_history
    history.insert(0, {'url': url[:60], 'verdict': verdict, 'score': score})
    st.session_state.scan_history = history[:5]

def render_history():
    if not st.session_state.get('scan_history'):
        return
    st.markdown('<p class="section-title">🕐 Recent Scans</p>', unsafe_allow_html=True)
    colors = {'CRITICAL':'#ff5252','HIGH':'#ff9800','MEDIUM':'#ffeb3b','LOW':'#69f0ae','SAFE':'#00e676'}
    for item in st.session_state.scan_history:
        c = colors.get(item['verdict'], '#aaa')
        st.markdown(
            f'<div class="history-item">'
            f'<span style="color:{c}">● {item["verdict"]}</span> '
            f'<span style="color:#aaa">{item["score"]:.3f}</span> — '
            f'<span style="color:#ccc">{item["url"]}</span>'
            f'</div>',
            unsafe_allow_html=True
        )


# ── Single URL analysis ────────────────────────────────────
def analyze_url(url, xgb_model, rf_model, explainer):
    with st.spinner("Analyzing..."):
        result   = hybrid_predict(url)
        features = extract_features(url)

    verdict = result['verdict']
    score   = result['score']
    cvss    = result['cvss']

    # Infer MITRE techniques from features if Layer 1 didn't catch anything
    mitre = result['mitre_techniques']
    if not mitre:
        mitre = infer_mitre_from_features(features, verdict)

    add_to_history(url, verdict, score)

    st.markdown("---")
    st.markdown('<p class="section-title">📊 Analysis Results</p>', unsafe_allow_html=True)
    st.code(url, language=None)

    # Verdict banner
    css = {'CRITICAL':'v-critical','HIGH':'v-high','MEDIUM':'v-medium','LOW':'v-low','SAFE':'v-safe'}
    icons = {'CRITICAL':'🔴','HIGH':'🟠','MEDIUM':'🟡','LOW':'🟢','SAFE':'✅'}
    st.markdown(
        f'<div class="verdict-box {css[verdict]}">{icons[verdict]} {verdict} RISK</div>',
        unsafe_allow_html=True
    )

    # Gauge
    render_gauge(score, cvss)

    # Plain English summary
    st.markdown('<p class="section-title">💡 What This Means</p>', unsafe_allow_html=True)

    plain_summaries = {
        'CRITICAL': "⚠️ This URL is almost certainly a phishing attack. Do NOT click this link. It is designed to steal your login credentials, banking details, or personal information. The AI detected this with near-100% confidence.",
        'HIGH':     "⚠️ This URL shows strong signs of being a phishing attack. It is very likely designed to deceive you. Avoid clicking and report it to your IT team or email provider.",
        'MEDIUM':   "⚠️ This URL has suspicious characteristics. It may or may not be harmful. Proceed with caution — do not enter any personal information until you verify the site is legitimate.",
        'LOW':      "ℹ️ This URL shows minor suspicious signals but is probably not an active phishing attack. The risk is low. If unsure, contact the sender through a separate channel to verify.",
        'SAFE':     "✅ No phishing indicators detected. This URL appears to be safe based on all 24 detection features. However, always remain cautious when entering personal information online."
    }

    st.markdown(f'<div class="plain-box">{plain_summaries[verdict]}</div>', unsafe_allow_html=True)

    # Jargon version (expandable)
    with st.expander("Technical explanation (for cybersecurity professionals)"):
        st.markdown(f'<div class="jargon-box">{result["explanation"]}</div>', unsafe_allow_html=True)
        st.markdown(f"**Layer 1 (Heuristic):** {result['layer1_verdict']} — Rules: {result['layer1_rules'] or 'None triggered'}")
        st.markdown(f"**Layer 2 (ML):** Score {result['layer2_score']} — XGBoost × 0.5001 + RF × 0.4999")
        st.markdown(f"**Layer 3 (Hybrid):** Escalated: {result['escalated']} | Trusted domain: {_is_trusted_domain(url)}")

    # MITRE + suspicious features
    st.markdown("---")
    col_mitre, col_feats = st.columns([3, 2])

    with col_mitre:
        st.markdown('<p class="section-title">🎯 Attack Techniques Detected</p>', unsafe_allow_html=True)

        if mitre:
            for t in mitre:
                info = MITRE_INFO.get(t, {'name': t, 'plain': '', 'jargon': ''})
                st.markdown(
                    f'<span class="badge-mitre">{t}</span> '
                    f'<span class="badge-mitre-desc">{info["name"]}</span>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<div class="plain-box" style="margin-top:4px; font-size:13px;">'
                    f'<b>Plain English:</b> {info["plain"]}</div>',
                    unsafe_allow_html=True
                )
                with st.expander(f"Technical detail — {t}"):
                    st.markdown(f"`{info['jargon']}`")
        else:
            st.markdown("No specific attack techniques identified. Verdict based on statistical URL patterns.")

        # Heuristic rules
        if result['layer1_rules']:
            st.markdown("**Heuristic Rules Triggered:**")
            for r in result['layer1_rules']:
                if r in RULES:
                    _, sev, _, desc = RULES[r]
                    st.markdown(f'<span class="badge-rule">⚡ {r}</span> {desc}', unsafe_allow_html=True)

    with col_feats:
        st.markdown('<p class="section-title">🔍 Top Suspicious Signals</p>', unsafe_allow_html=True)

        top_feats = get_top_suspicious_features(features, verdict)
        if top_feats:
            for f in top_feats:
                st.markdown(
                    f'<div class="feat-card">'
                    f'<b style="color:#ff8a80">{f["feature"]}</b> = {f["value"]}<br>'
                    f'<span style="color:#ccc; font-size:13px;">{f["plain"]}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                with st.expander(f"Technical: {f['feature']}"):
                    st.markdown(f"`{f['jargon']}`")
        else:
            st.markdown(
                '<div class="feat-card-safe">No strongly suspicious individual features detected. '
                'The ML ensemble detected subtle combined patterns.</div>',
                unsafe_allow_html=True
            )

    # SHAP + Feature values
    st.markdown("---")
    left, right = st.columns([3, 2])

    with left:
        st.markdown('<p class="section-title">📈 Feature Contribution Chart (SHAP)</p>',
                    unsafe_allow_html=True)
        st.caption("Shows WHICH features pushed the AI toward phishing (red) or legitimate (blue)")
        try:
            fig = shap_waterfall(features, xgb_model, explainer)
            st.pyplot(fig, use_container_width=True)
            plt.close()
            st.caption("🔴 Red bars = push toward PHISHING  |  🔵 Blue bars = push toward LEGITIMATE")
        except Exception as e:
            st.warning(f"Chart unavailable: {e}")

    with right:
        st.markdown('<p class="section-title">📋 All 24 Detection Features</p>',
                    unsafe_allow_html=True)

        groups = {
            '🔩 Structural': ['url_length','qty_dot','qty_hyphen','qty_slash',
                               'qty_at','qty_question','qty_equal','qty_percent',
                               'domain_length','tld_length'],
            '📊 Statistical': ['url_entropy','domain_entropy','digit_ratio',
                                'alpha_ratio','char_continuation_rate'],
            '🔍 Content': ['has_https','has_shortener','has_login_keyword',
                            'is_ip','brand_in_path_not_domain'],
            '⚡ Advanced': ['recursive_decode_depth','idn_homograph_flag',
                             'levenshtein_min','tld_risk_score'],
        }

        for group, feat_list in groups.items():
            with st.expander(group, expanded=(group == '⚡ Advanced')):
                rows = [{'Feature': f, 'Value': features.get(f, 0)} for f in feat_list]
                st.dataframe(pd.DataFrame(rows), use_container_width=True,
                             hide_index=True, height=min(len(rows)*38+38, 300))

    # Layer breakdown
    st.markdown("---")
    st.markdown('<p class="section-title">🏗️ 3-Layer Detection Breakdown</p>',
                unsafe_allow_html=True)
    l1, l2, l3 = st.columns(3)

    with l1:
        st.markdown("**Layer 1 — Rule Engine**")
        st.caption("Instant detection of known attack patterns")
        color = "🔴" if result['layer1_verdict']=='HIGH_RISK' else \
                "🟡" if result['layer1_verdict']=='SUSPICIOUS' else "🟢"
        st.markdown(f"{color} **{result['layer1_verdict']}**")
        if result['layer1_rules']:
            for r in result['layer1_rules']:
                st.markdown(f"- `{r}`")
        else:
            st.markdown("- No smoking-gun patterns found")
            st.caption("URL passed to AI for deeper analysis")

    with l2:
        st.markdown("**Layer 2 — AI Detection**")
        st.caption("Machine learning ensemble decision")
        if result['layer2_score'] is not None:
            s = result['layer2_score']
            c = "🔴" if s>0.7 else "🟡" if s>0.4 else "🟢"
            st.markdown(f"{c} Confidence: **{s:.4f}**")
            st.markdown("- XGBoost (weight: 0.5001)")
            st.markdown("- Random Forest (weight: 0.4999)")
            st.caption("Weights determined by F1 score comparison")
        else:
            st.markdown("⏭️ Bypassed — Layer 1 already decided")

    with l3:
        st.markdown("**Layer 3 — Final Decision**")
        st.caption("Combines all signals intelligently")
        if result['escalated']:
            st.markdown("⚠️ **ESCALATED**")
            st.caption("Rules said suspicious, AI said safe — system chose to escalate (do not average out real threats)")
        elif _is_trusted_domain(url):
            st.markdown("🏦 **Trusted Domain**")
            st.caption("Known legitimate site — score capped to prevent false alarm")
        else:
            st.markdown("✅ Standard weighted combination")
        st.markdown(f"**Final Score:** {result['score']:.4f}")
        st.markdown(f"**CVSS:** {result['cvss']} / 10.0")


# ── Batch analysis ─────────────────────────────────────────
def batch_analysis(xgb_model, rf_model, explainer):
    st.markdown("---")
    st.markdown('<p class="section-title">📁 Batch URL Scanner</p>', unsafe_allow_html=True)
    st.markdown("Upload a text file (one URL per line) or CSV file with a `url` column. The system will scan all URLs and produce a downloadable report.")

    uploaded = st.file_uploader(
        "Upload URL file",
        type=['txt', 'csv'],
        help="Text file: one URL per line. CSV file: must have a column named 'url'"
    )

    if uploaded is None:
        return

    # Parse file
    try:
        if uploaded.name.endswith('.csv'):
            df_input = pd.read_csv(uploaded)
            if 'url' not in df_input.columns:
                st.error("CSV must have a column named 'url'")
                return
            urls = df_input['url'].dropna().tolist()
        else:
            content = uploaded.read().decode('utf-8', errors='ignore')
            urls = [u.strip() for u in content.splitlines() if u.strip().startswith('http')]
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return

    if not urls:
        st.warning("No valid URLs found in file.")
        return

    st.info(f"Found {len(urls):,} URLs. Starting scan...")

    # Scan with progress bar
    progress = st.progress(0)
    status = st.empty()
    results = []

    for i, url in enumerate(urls):
        try:
            result   = hybrid_predict(url)
            features = extract_features(url)
            mitre    = result['mitre_techniques']
            if not mitre:
                mitre = infer_mitre_from_features(features, result['verdict'])

            results.append({
                'URL': url[:120],
                'Verdict': result['verdict'],
                'Score': result['score'],
                'CVSS': result['cvss'],
                'Layer1': result['layer1_verdict'],
                'ML_Score': result['layer2_score'],
                'MITRE': ', '.join(mitre),
                'Escalated': result['escalated'],
            })
        except Exception as e:
            results.append({
                'URL': url[:120], 'Verdict': 'ERROR', 'Score': None,
                'CVSS': None, 'Layer1': 'ERROR', 'ML_Score': None,
                'MITRE': str(e)[:50], 'Escalated': False
            })

        progress.progress((i+1)/len(urls))
        if (i+1) % 10 == 0:
            status.text(f"Scanned {i+1}/{len(urls)} URLs...")

    status.text("Scan complete!")
    df_results = pd.DataFrame(results)

    # Summary stats
    st.markdown("**Scan Summary:**")
    verdict_counts = df_results['Verdict'].value_counts()
    cols = st.columns(5)
    for i, (v, c) in enumerate(verdict_counts.items()):
        color = {'CRITICAL':'#ff5252','HIGH':'#ff9800','MEDIUM':'#ffeb3b',
                 'LOW':'#69f0ae','SAFE':'#00e676','ERROR':'#aaa'}.get(v, '#aaa')
        with cols[min(i, 4)]:
            st.markdown(f'<div style="text-align:center; color:{color}; font-size:24px; font-weight:bold">{c}</div>'
                        f'<div style="text-align:center; font-size:12px; color:#aaa">{v}</div>',
                        unsafe_allow_html=True)

    # Results table
    st.markdown("**Detailed Results:**")
    st.dataframe(df_results, use_container_width=True, hide_index=True)

    # Download button
    csv_data = df_results.to_csv(index=False)
    st.download_button(
        label="⬇️ Download Results CSV",
        data=csv_data,
        file_name="phishlens_batch_results.csv",
        mime="text/csv"
    )


# ── Main ───────────────────────────────────────────────────
def main():
    init_history()
    sidebar()

    st.title("PhishLens")
    st.caption("Hybrid Phishing Detection System — Sunway University CP2 Capstone")
    st.markdown("---")

    # Load models
    try:
        xgb_model, rf_model = load_models()
        explainer = load_explainer(xgb_model)
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.info("Run: python src/model_trainer.py --augmented")
        return

    # Tabs
    tab1, tab2 = st.tabs(["🔍 Single URL Analysis", "📁 Batch Scan"])

    with tab1:
        st.markdown('<p class="section-title">🔍 Analyze a URL</p>', unsafe_allow_html=True)
        col_input, col_btn = st.columns([6, 1])
        with col_input:
            url_input = st.text_input(
                "URL", placeholder="Paste any URL here — https://example.com",
                label_visibility="collapsed"
            )
        with col_btn:
            analyze = st.button("Analyze", type="primary", use_container_width=True)

        # Quick tests with descriptive names
        st.markdown("**Quick tests — click to demo:**")
        qc = st.columns(4)
        quick = [
            ("✅ Legitimate (Google)",
             "https://www.google.com",
             "Tests a real legitimate URL — should show SAFE"),
            ("✅ Malaysian Bank (Maybank)",
             "https://www.maybank2u.com.my/login",
             "Tests a real Malaysian banking login — should show SAFE"),
            ("🔴 Brand Spoof + Suspicious TLD",
             "https://paypal-secure-login.evil.tk/account/verify?id=12345",
             "Classic phishing: brand name in subdomain + .tk TLD — should show CRITICAL"),
            ("🔴 AiTM Kit (m365 pattern)",
             "https://m365-login.suspicious.tk/auth",
             "Matches known adversary-in-the-middle phishing kit pattern — should show CRITICAL"),
        ]

        chosen = None
        for i, (label, url, tooltip) in enumerate(quick):
            with qc[i]:
                if st.button(label, use_container_width=True, help=tooltip, key=f"q{i}"):
                    chosen = url

        target_url = chosen or (url_input if analyze and url_input else None)

        if not target_url:
            st.markdown("---")
            st.info("Paste a URL above and click **Analyze**, or click one of the quick test buttons to see a live demo.")
            render_history()
            return

        analyze_url(target_url, xgb_model, rf_model, explainer)
        st.markdown("---")
        render_history()

    with tab2:
        batch_analysis(xgb_model, rf_model, explainer)


if __name__ == "__main__":
    main()
