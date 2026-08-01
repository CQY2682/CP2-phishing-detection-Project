<h1 align="center">🛡️ PhishLens</h1>

<p align="center">
  <b>A lightweight, explainable, hybrid machine-learning framework for phishing-URL detection.</b><br>
  MITRE-mapped heuristics + a tree-based ML ensemble, using <b>lexical URL features only</b> — no page content, real-time by design.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/scikit--learn-1.5-F7931E?logo=scikitlearn&logoColor=white">
  <img src="https://img.shields.io/badge/XGBoost-2.1-016?logo=xgboost&logoColor=white">
  <img src="https://img.shields.io/badge/SHAP-explainable-8b5cf6">
  <img src="https://img.shields.io/badge/Flask-web%20dashboard-000000?logo=flask&logoColor=white">
  <img src="https://img.shields.io/badge/F1-0.9976-2de2b0">
  <img src="https://img.shields.io/badge/status-complete-brightgreen">
</p>

<p align="center">
  <b>Author:</b> Cheah Qi Yang (22095483) &nbsp;·&nbsp;
  <b>Supervisor:</b> Dr Mohd Firdaus Roslan<br>
  BSc Information Technology (Networking &amp; Security) (Hons), Sunway University &nbsp;·&nbsp; Capstone Project 2
</p>

---

## Table of Contents

- [Highlights](#highlights)
- [How It Works](#how-it-works)
- [The Dashboards](#the-dashboards)
- [Quick Start](#quick-start)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [The 24 Lexical Features](#the-24-lexical-features)
- [Heuristic Rules (Layer 1)](#heuristic-rules-layer-1)
- [Dataset](#dataset)
- [Limitations](#limitations)
- [Tech Stack](#tech-stack)
- [License &amp; Acknowledgements](#license--acknowledgements)

---

## Highlights

Phishing remains a leading cause of credential theft and financial fraud. Blacklists are reactive and miss zero-day URLs; deep-learning detectors are accurate but heavy and hard to interpret. **PhishLens targets the middle ground: high accuracy, low latency, and explainable decisions, using only features extracted from the URL string.**

| | |
|---|---|
| **Accuracy** | 99.80% |
| **F1 / ROC-AUC / MCC** | 0.9976 / 0.9990 / 0.9959 |
| **Adversarial robustness** | 100% retention across 5 mutation types (0 evasions) |
| **Blind hold-out (OpenPhish)** | 100% detection on 300 unseen real phishing URLs |
| **Explainability** | Per-URL SHAP attributions + plain-English reasons |
| **Features** | 24 lexical features, no page content downloaded |

> These numbers describe a deliberately easy benchmark; the report is explicit about that. The **robustness and blind hold-out tests** are the real evidence of practical capability, not the clean-set metrics alone.

---

## How It Works

A URL is classified through a **three-layer defense-in-depth pipeline**:

```
        Raw URL
           │
           ▼
┌─────────────────────────────┐
│ Layer 1 — Heuristic Engine  │  8 MITRE-mapped rules.
│ (src/heuristic_engine.py)   │  A HIGH_RISK rule → instant verdict, ML bypassed.
└─────────────────────────────┘
           │ (clean / suspicious → passes on)
           ▼
┌─────────────────────────────┐
│ Layer 2 — ML Ensemble       │  XGBoost + Random Forest.
│ (src/model_trainer.py)      │  F1-justified weights (0.5001 / 0.4999).
└─────────────────────────────┘
           │
           ▼
┌─────────────────────────────┐
│ Layer 3 — Hybrid Scorer     │  Disagreement Escalation + trusted-domain
│ (src/hybrid_scorer.py)      │  whitelist. Outputs CVSS-inspired severity.
└─────────────────────────────┘
           │
           ▼
   Severity + explanation  →  web dashboard (SHAP, plain English)
```

**Key design choices**

- **Fail-fast (Layer 1):** obvious threats (IP-based URLs, punycode, embedded credentials) are caught instantly, without invoking the ML models.
- **Disagreement Escalation (Layer 3):** when the heuristic flags a URL suspicious but the ML scores it safe, the verdict is *escalated*, not averaged — favouring caution.
- **Trusted-domain whitelist (Layer 3):** 30 known-legitimate domains (e.g. `maybank2u.com.my`, `sunway.edu.my`) only ever *lower* a score to suppress false positives; the whitelist can never create a detection.

---

## The Dashboards

PhishLens ships with **two interfaces** over the same detection engine (both call `hybrid_predict`):

### 1. Web dashboard (primary) — `dashboard/server.py`

A dark security-console UI (Flask backend + static front end). Features:

- Paste-any-URL analysis with a severity ring and 0–10 CVSS-inspired meter
- Plain-English **"Why this verdict"** reasons + a **recommended action** line
- Live **three-layer breakdown** and MITRE ATT&CK technique badges
- On-demand **SHAP** feature-contribution chart
- A **Threat Library** of illustrative attack specimens, an **Overview** metrics page, a live **recent-scans** list, session counters, and a downloadable plain-text report

### 2. Streamlit prototype (fallback) — `dashboard/app.py`

The original Streamlit UI, kept as a lightweight backup with the same detection output.

> **Screenshots** (add your own to a `docs/` folder):
> `docs/analyze.png` · `docs/overview.png` · `docs/threat-library.png`

---

## Quick Start

```bash
# 0. Activate the virtual environment
.\.venv\Scripts\Activate.ps1          # Windows PowerShell
# source .venv/bin/activate           # macOS / Linux

# 1. Install dependencies
pip install -r requirements.txt

# ── To reproduce the models & results from scratch ──────────────
python src/preprocessing.py           # label flip, dedup, stratified split
python src/model_trainer.py --augmented
python src/evaluation.py              # metrics, ROC, confusion, McNemar
python src/train_val_test.py          # overfitting analysis
python src/learning_curve.py          # learning-curve convergence
python src/hyperparameter_tuning.py   # grid search (48 combos, 5-fold CV)
python src/adversarial_mutator.py     # adversarial robustness
python src/phishtank_holdout.py       # OpenPhish blind hold-out
python src/leakage_check.py           # data-integrity audit
python src/honest_eval.py             # collision-free re-evaluation

# ── To run the dashboards (models must already exist) ───────────
python dashboard/server.py            # web dashboard  → open http://127.0.0.1:8000
streamlit run dashboard/app.py        # Streamlit fallback
```

> Run all commands from the **project root**. On some machines use `py` instead of `python`.
> The web dashboard is local-only (no internet, no account, no cost).

---

## Results

**Test set (47,074 URLs) — Hybrid Ensemble**

| Metric | Score |
|---|---|
| Accuracy | 0.9980 |
| Precision | 0.9998 |
| Recall | 0.9955 |
| F1 | 0.9976 |
| ROC-AUC | 0.9990 |
| MCC | 0.9959 |

**Robustness &amp; generalization**

- **Adversarial testing:** 100% detection retention across 5 mutation types (zero evasions).
- **OpenPhish blind hold-out:** 100% detection on 300 unseen real phishing URLs; 4 caught by Layer 1, 296 by Layer 2, 0 escalations.
- **Overfitting:** train/val/test F1 gap < 0.002; learning curves converge.
- **Baselines:** stratified random F1 ≈ 0.43; logistic-regression baseline F1 = 0.9947 — a strong linear baseline confirms the dataset is easily separable.
- **Data-integrity audit:** ~49.8% of test rows share a feature vector with training (feature collision, intrinsic to compact lexical features). On the **collision-free subset**, F1 = 0.9977 — unchanged, confirming performance is not memorization.

---

## Repository Structure

| Path | Purpose | Produces |
|---|---|---|
| `src/feature_extractor.py` | Extracts the 24 lexical features (incl. Malay login keywords) | feature vectors |
| `src/preprocessing.py` | Label flip (raw PhiUSIIL is reversed), dedup, stratified 80/20 split | `data/X_*.csv`, `data/y_*.csv` |
| `src/heuristic_engine.py` | Layer 1 — 8 MITRE ATT&CK-mapped rules | rule verdicts |
| `src/augment_training.py` | Adds 124 curated real-world URLs to reduce homepage bias | augmented split |
| `src/model_trainer.py` | Layer 2 — trains XGBoost + Random Forest; F1-justified weights (`--augmented`) | `models/*.joblib`, `reports/training_metrics.json` |
| `src/hybrid_scorer.py` | Layer 3 — Disagreement Escalation, trusted-domain whitelist, CVSS severity | final classification |
| `src/adversarial_mutator.py` | 5 mutation types to stress-test robustness | `reports/adversarial_results.csv` |
| `src/phishtank_holdout.py` | Blind hold-out on the OpenPhish feed | `reports/openphish_holdout_results.csv` |
| `src/evaluation.py` | Metrics, ROC, confusion matrices, baselines, McNemar's test | `reports/full_evaluation.json`, figures |
| `src/hyperparameter_tuning.py` | Grid search (48 combos, 5-fold CV) | `reports/hyperparameter_results.json` |
| `src/learning_curve.py` | Learning-curve convergence (overfitting check) | figures |
| `src/train_val_test.py` | 64/16/20 split overfitting analysis | `reports/train_val_test_results.json` |
| `src/leakage_check.py` | Feature-space overlap / data-integrity audit | `reports/leakage_check.json` |
| `src/honest_eval.py` | Re-evaluates on the collision-free subset | `reports/collision_analysis.json` |
| `dashboard/server.py` | **Web dashboard** — Flask backend + `/api/analyze`, `/api/shap` | serves `static/index.html` |
| `dashboard/static/index.html` | **Web dashboard** — front end (analyze, SHAP, threat library, overview) | interactive UI |
| `dashboard/app.py` | Streamlit prototype (fallback UI) | interactive app |
| `reports/figures/` | All generated figures (ROC, confusion, learning curves, class separation) | `.png` |
| `reports/` | All metric outputs | `.json` / `.csv` |
| `tests/` | Unit / integration tests | test results |

> **Note:** `data/` and `models/` are git-ignored (large files). They are regenerated by the preprocessing and training scripts.

---

## The 24 Lexical Features

Selected by a **knowledge-driven method** (MITRE ATT&CK + literature), validated post-training by XGBoost feature importance.

- **Structural (1–10):** url_length, qty_dot, qty_hyphen, qty_slash, qty_at, qty_question, qty_equal, qty_percent, domain_length, tld_length
- **Statistical (11–15):** url_entropy, domain_entropy, digit_ratio, alpha_ratio, char_continuation_rate
- **Content (16–20):** has_https, has_shortener, has_login_keyword, is_ip, brand_in_path_not_domain
- **Advanced / MITRE-aligned (21–24):** recursive_decode_depth (T1027.001), idn_homograph_flag (T1036.007), levenshtein_min (T1566.002), tld_risk_score (T1583.001)

---

## Heuristic Rules (Layer 1)

| Rule | Triggers on | MITRE technique |
|---|---|---|
| `ip_url` | IP address as domain | T1071.001 |
| `idn_punycode` | `xn--` punycode encoding | T1036.007 |
| `multi_encoded` | URL encoded 2+ times | T1027.001 |
| `aitm_pattern` | adversary-in-the-middle patterns (e.g. `m365-login`) | T1557 |
| `brand_spoof_sus_tld` | brand in path + high TLD risk | T1583.001 + T1566.002 |
| `ip_with_login` | IP + login keywords | T1071.001 + T1566.002 |
| `embedded_credentials` | `@` symbol in URL | T1566.002 |
| `typosquat` | Levenshtein distance 1 from a known brand | T1566.002 |

> Seven rules return an instant **HIGH_RISK** verdict (ML bypassed). `embedded_credentials` is **SUSPICIOUS**, so it passes to Layer 2 and can trigger Disagreement Escalation.

---

## Dataset

- **Primary:** PhiUSIIL Phishing URL Dataset (Prasad &amp; Chandra, 2023), UCI ML Repository — 235,795 rows. Only the `url` and `label` columns are used; the 24 features are extracted by this project's own pipeline.
- **Label note:** raw PhiUSIIL labels are reversed (0 = phishing, 1 = legitimate); `preprocessing.py` flips them so that 1 = phishing.
- After deduplication: 235,370 unique rows → 188,296 train / 47,074 test (stratified).
- **Blind hold-out:** OpenPhish Community Feed (used because PhishTank public access was disabled).

---

## Limitations

- **Lexical-only:** the system reads the URL string, never page content — fast and privacy-preserving, but blind to rendered-page behaviour. A long, structurally complex *legitimate* URL can therefore be a false positive; this is a documented consequence of the dataset bias, mitigated by the trusted-domain whitelist and slated for a future content-inspection layer.
- **Feature collision:** compact lexical features map distinct URLs to identical vectors; documented and controlled for via the collision-free evaluation.
- **Dataset characteristics:** PhiUSIIL legitimate URLs are largely short homepages and 100% HTTPS, which inflates a few features; augmentation, the whitelist, and the independent OpenPhish hold-out address this.

---

## Tech Stack

`Python 3.11` · `pandas` · `numpy` · `scikit-learn` · `xgboost` · `shap` · `flask` · `streamlit` · `statsmodels` · `tldextract` · `python-Levenshtein` · `joblib` · `matplotlib` · `seaborn`

---

## License &amp; Acknowledgements

Released under the **MIT License** — see `LICENSE`.

Built as a Capstone Project 2 at **Sunway University**. Dataset courtesy of the **UCI Machine Learning Repository** (PhiUSIIL). Attack-technique mappings follow the **MITRE ATT&CK** framework. Blind hold-out data from the **OpenPhish Community Feed**.

> This README documents the system as built. Evaluation is complete and the architecture (24 features, three-layer hybrid, Disagreement Escalation) is finalized.