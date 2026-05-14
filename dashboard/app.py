"""
CP2 Phishing Detection Dashboard
Author: Cheah Qi Yang (22095483)
Run: streamlit run dashboard/app.py
"""

import sys, os
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from feature_extractor import extract_features, FEATURE_NAMES
from hybrid_scorer import hybrid_predict, _is_trusted_domain, TRUSTED_DOMAINS
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
/* Hide Streamlit branding */
#MainMenu, footer { visibility: hidden; }

/* Main background */
.block-container { padding: 2rem 3rem; max-width: 1200px; }

/* Verdict banners */
.verdict-box {
    padding: 16px 24px; border-radius: 10px;
    font-size: 22px; font-weight: 700;
    text-align: center; letter-spacing: 1px;
    margin: 12px 0;
}
.v-critical { background: linear-gradient(90deg,#b00020,#ff1744); color:#fff; }
.v-high     { background: linear-gradient(90deg,#e65100,#ff6d00); color:#fff; }
.v-medium   { background: linear-gradient(90deg,#f57f17,#ffca28); color:#111; }
.v-low      { background: linear-gradient(90deg,#2e7d32,#66bb6a); color:#fff; }
.v-safe     { background: linear-gradient(90deg,#1b5e20,#43a047); color:#fff; }

/* MITRE badge */
.badge-mitre {
    display:inline-block; background:#1a237e; color:#fff;
    padding:3px 12px; border-radius:20px;
    font-size:12px; font-weight:600; margin:3px;
}
/* Rule badge */
.badge-rule {
    display:inline-block; background:#b71c1c; color:#fff;
    padding:3px 12px; border-radius:20px;
    font-size:12px; font-weight:600; margin:3px;
}
/* Info card */
.card {
    background:#16213e; border:1px solid #0f3460;
    border-radius:8px; padding:16px 20px; margin:8px 0;
    border-left: 3px solid #4fc3f7;
}
/* Section header */
.section-title {
    font-size:16px; font-weight:700;
    color:#90caf9; margin:16px 0 8px 0;
    border-bottom:1px solid #333; padding-bottom:6px;
}
/* Metric label fix */
[data-testid="stMetricLabel"] { font-size:13px !important; }
</style>
""", unsafe_allow_html=True)


# ── Model loader ───────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    xgb = joblib.load('models/xgboost_model.joblib')
    rf  = joblib.load('models/rf_model.joblib')
    return xgb, rf

@st.cache_resource(show_spinner="Loading SHAP explainer...")
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
    plt.title("Feature Contributions", fontsize=12, pad=10)
    plt.tight_layout()
    return fig


# ── Sidebar ────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown("## PhishLens")
        st.markdown("*URL Threat Intelligence*")
        st.markdown("---")
        st.markdown("**Project**")
        st.markdown("Hybrid ML Framework for Phishing Detection")
        st.markdown("👤 Cheah Qi Yang (22095483)")
        st.markdown("🏫 Sunway University")
        st.markdown("👨‍🏫 Dr Mohd Firdaus Roslan")
        st.markdown("---")
        st.markdown("**Architecture**")
        st.markdown("- **Layer 1** — Heuristic Engine (8 rules)\n- **Layer 2** — XGBoost + Random Forest\n- **Layer 3** — Hybrid Scorer + Escalation")
        st.markdown("---")
        st.markdown("**Severity Scale**")
        st.markdown("| Score | Level |\n|-------|-------|\n| ≥0.90 | 🔴 CRITICAL |\n| ≥0.70 | 🟠 HIGH |\n| ≥0.40 | 🟡 MEDIUM |\n| ≥0.20 | 🟢 LOW |\n| <0.20 | ✅ SAFE |")
        st.markdown("---")
        st.markdown("**2026 Threat Landscape**")
        st.markdown("- Tycoon 2FA takedown (Jan 2026)\n- Starkiller AiTM kit\n- Mamba 2FA successor\n- HTTPS laundering via Let's Encrypt")


# ── Verdict display ────────────────────────────────────────
def show_verdict(result):
    v = result['verdict']
    icons = {'CRITICAL':'🔴','HIGH':'🟠','MEDIUM':'🟡','LOW':'🟢','SAFE':'✅'}
    css   = {'CRITICAL':'v-critical','HIGH':'v-high','MEDIUM':'v-medium',
             'LOW':'v-low','SAFE':'v-safe'}
    st.markdown(
        f'<div class="verdict-box {css[v]}">'
        f'{icons[v]} {v} RISK'
        f'</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Phishing Score", f"{result['score']:.3f}")
    c2.metric("CVSS", f"{result['cvss']} / 10.0")
    layer1 = result['layer1_verdict']
    c3.metric("Layer 1", "⚠️ Escalated" if result['escalated'] else layer1)


# ── Main ───────────────────────────────────────────────────
def main():
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
        return

    # Input row
    st.markdown('<p class="section-title">🔍 Analyze a URL</p>', unsafe_allow_html=True)
    col_input, col_btn = st.columns([6, 1])
    with col_input:
        url_input = st.text_input("URL", placeholder="https://example.com",
                                  label_visibility="collapsed")
    with col_btn:
        analyze = st.button("Analyze", type="primary", use_container_width=True)

    # Quick tests
    st.markdown("**Quick tests:**")
    qc = st.columns(4)
    quick = [
        ("✅ Google",   "https://www.google.com"),
        ("✅ Maybank",  "https://www.maybank2u.com.my/login"),
        ("🔴 Phishing", "https://paypal-secure-login.evil.tk/account/verify?id=12345"),
        ("🔴 AiTM",    "https://m365-login.suspicious.tk/auth"),
    ]
    chosen = None
    for i, (label, url) in enumerate(quick):
        with qc[i]:
            if st.button(label, use_container_width=True, key=f"q{i}"):
                chosen = url

    target_url = chosen or (url_input if analyze and url_input else None)

    if not target_url:
        st.info("Enter a URL or click a quick test above.")
        return

    # Run pipeline
    with st.spinner("Running analysis..."):
        result  = hybrid_predict(target_url)
        features = extract_features(target_url)

    st.markdown("---")

    # ── RESULTS ──────────────────────────────────────────
    st.markdown('<p class="section-title">📊 Results</p>', unsafe_allow_html=True)
    st.code(target_url, language=None)
    show_verdict(result)

    # Explanation
    st.markdown('<p class="section-title">💡 Explanation</p>', unsafe_allow_html=True)
    st.markdown(f'<div class="card">{result["explanation"]}</div>',
                unsafe_allow_html=True)

    # MITRE + rules
    st.markdown('<p class="section-title">🎯 Attack Patterns</p>',
                unsafe_allow_html=True)

    mitre = result['mitre_techniques']
    rules = result['layer1_rules']

    if mitre:
        badges = " ".join(f'<span class="badge-mitre">{t}</span>' for t in mitre)
        st.markdown(f"**MITRE ATT&CK:** {badges}", unsafe_allow_html=True)
    else:
        st.markdown("**MITRE ATT&CK:** No techniques triggered")

    if rules:
        for r in rules:
            if r in RULES:
                _, sev, mids, desc = RULES[r]
                st.markdown(
                    f'<span class="badge-rule">⚠️ {r}</span> — {desc}',
                    unsafe_allow_html=True
                )
    else:
        st.markdown("No heuristic rules triggered — verdict from ML ensemble.")

    st.markdown("---")

    # ── SHAP + FEATURES side by side ─────────────────────
    left, right = st.columns([3, 2])

    with left:
        st.markdown('<p class="section-title">📈 SHAP Waterfall</p>',
                    unsafe_allow_html=True)
        try:
            fig = shap_waterfall(features, xgb_model, explainer)
            st.pyplot(fig, use_container_width=True)
            plt.close()
            st.caption("🔴 Red = pushes toward phishing  |  🔵 Blue = pushes toward legitimate")
        except Exception as e:
            st.warning(f"SHAP unavailable: {e}")

    with right:
        st.markdown('<p class="section-title">🔢 Feature Values</p>',
                    unsafe_allow_html=True)
        feat_df = pd.DataFrame([
            {'Feature': k, 'Value': v}
            for k, v in features.items()
        ])
        st.dataframe(feat_df, use_container_width=True,
                     hide_index=True, height=420)

    # ── Layer breakdown ───────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-title">🏗️ Layer-by-Layer Breakdown</p>',
                unsafe_allow_html=True)

    l1, l2, l3 = st.columns(3)

    with l1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Layer 1 — Heuristic Engine**")
        color = "🔴" if result['layer1_verdict']=='HIGH_RISK' else \
                "🟡" if result['layer1_verdict']=='SUSPICIOUS' else "🟢"
        st.markdown(f"{color} **{result['layer1_verdict']}**")
        if rules:
            for r in rules: st.markdown(f"- `{r}`")
        else:
            st.markdown("- No rules triggered")
        st.markdown('</div>', unsafe_allow_html=True)

    with l2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Layer 2 — ML Ensemble**")
        if result['layer2_score'] is not None:
            s = result['layer2_score']
            c = "🔴" if s>0.7 else "🟡" if s>0.4 else "🟢"
            st.markdown(f"{c} Score: **{s:.4f}**")
            st.markdown("- XGBoost × 0.5001")
            st.markdown("- Random Forest × 0.4999")
        else:
            st.markdown("⏭️ Bypassed by Layer 1")
        st.markdown('</div>', unsafe_allow_html=True)

    with l3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Layer 3 — Hybrid Scorer**")
        if result['escalated']:
            st.markdown("⚠️ **Disagreement Escalation**")
            st.markdown("Heuristic: SUSPICIOUS")
            st.markdown("ML: SAFE → **ESCALATED**")
        elif _is_trusted_domain(target_url):
            st.markdown("🏦 Trusted domain")
            st.markdown("Score capped at SAFE")
        else:
            st.markdown("✅ Weighted average")
        st.markdown(f"**Final:** {result['score']:.4f}")
        st.markdown(f"**CVSS:** {result['cvss']} / 10.0")
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()