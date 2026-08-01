"""
PhishLens web backend (Flask).
Serves the front-end page and exposes the detection model as a small local API.

Run from the PROJECT ROOT:
    python dashboard/server.py
Then open http://127.0.0.1:8000 in your browser.

This is 100% local. No internet, no account, no cost.
"""

import os
import sys

# --- make paths work no matter where you launch from -------------------------
HERE = os.path.dirname(os.path.abspath(__file__))          # .../dashboard
ROOT = os.path.abspath(os.path.join(HERE, ".."))           # project root
os.chdir(ROOT)                                             # so 'models/...' etc. resolve
sys.path.insert(0, os.path.join(ROOT, "src"))              # so we can import your modules

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory

# your existing detection pipeline, imported unchanged
from hybrid_scorer import hybrid_predict
from feature_extractor import extract_features, FEATURE_NAMES

app = Flask(
    __name__,
    static_folder=os.path.join(HERE, "static"),
    static_url_path="",
)

# warm the models once at startup so the first request is not slow
print("Loading models, please wait...")
try:
    _ = hybrid_predict("https://www.google.com")
    print("Models loaded. Ready.")
except Exception as e:
    print("WARNING: model warm-up failed:", e)

# SHAP explainer is loaded lazily, only the first time someone clicks "Explain"
_explainer = None


def _get_explainer():
    global _explainer
    if _explainer is None:
        import joblib
        import shap
        xgb = joblib.load("models/xgboost_model.joblib")
        _explainer = shap.TreeExplainer(xgb)
    return _explainer


def _clean(o):
    """Convert numpy types so Flask can turn the result into JSON."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json(force=True, silent=True) or {}
    url = str(data.get("url", "")).strip()
    if not url:
        return jsonify({"error": "no url provided"}), 400
    try:
        result = hybrid_predict(url)
        try:
            result["features"] = {k: float(v) for k, v in extract_features(url).items()}
        except Exception:
            result["features"] = {}
        return jsonify(_clean(result))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/shap", methods=["POST"])
def api_shap():
    """Return per-feature SHAP contributions for one URL (on demand)."""
    data = request.get_json(force=True, silent=True) or {}
    url = str(data.get("url", "")).strip()
    if not url:
        return jsonify({"error": "no url provided"}), 400
    try:
        feats = extract_features(url)
        X = pd.DataFrame([feats])[FEATURE_NAMES]
        explainer = _get_explainer()
        sv = explainer(X)
        vals = sv.values[0]
        data_vals = sv.data[0]
        base = sv.base_values[0]
        items = []
        for name, fval, sval in zip(FEATURE_NAMES, data_vals, vals):
            items.append({"feature": name, "value": float(fval), "shap": float(sval)})
        items.sort(key=lambda d: abs(d["shap"]), reverse=True)
        return jsonify({"base_value": float(base), "features": items[:12]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("PhishLens running at  http://127.0.0.1:8000")
    app.run(host="127.0.0.1", port=8000, debug=False)