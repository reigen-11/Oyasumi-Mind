import os
import json
import pickle
import traceback
from datetime import datetime
from functools import wraps

import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request, jsonify

from utils.logger import logging
from utils.custom_exceptions import CustomException


# ---- Config + app ----
CONFIG_PATH = os.environ.get("CONFIG_PATH", "config.json")
with open(CONFIG_PATH, "r") as cf:
    config = json.load(cf)

app = Flask(__name__)
app.config.update({
    "ENV": os.environ.get("FLASK_ENV", "production"),
    # DEBUG controlled explicitly by FLASK_DEBUG env var below
    "DEBUG": False
})
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

logger = logging.getLogger("mental_wellness_app")

# allow runtime log level override
_log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
try:
    logger.setLevel(getattr(logging, _log_level))
except Exception:
    logger.setLevel(logging.INFO)


MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    config.get("model_training_settings", {}).get("trained_model_path")
)
ENCODER_PATH = os.environ.get(
    "ENCODER_PATH",
    config.get("data_ingestion_settings", {}).get("label_encoder_path")
)


# ---- utility functions ----
def safe_load_pickle(path, name="object"):
    """Load pickle but never raise — return None on any error and log it.
    This prevents a failing model load from crashing the process in production.
    """
    if not path:
        logger.warning("No path provided for %s; returning None", name)
        return None
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("Loaded %s from %s", name, path)
        return obj
    except FileNotFoundError:
        logger.warning("File not found for %s: %s", name, path)
        return None
    except Exception as e:
        logger.exception("Failed to load %s from %s: %s", name, path, e)
        return None


# load model & encoder (non-fatal if missing)
model = safe_load_pickle(MODEL_PATH, name="model")
encoder = safe_load_pickle(ENCODER_PATH, name="encoder")


FEATURES = config["column_settings"]["features"]
CATEGORICAL = config["column_settings"].get("categorical", [])
NUMERIC = config["column_settings"].get("numeric", [])
TARGET = config["column_settings"].get("target", "mental_wellness_index_0_100")

CLEANING_RULES = config.get("cleaning_rules", {
    "age": [10, 90],
    "sleep_hours": [0, 24],
    "stress_level_0_10": [0, 10],
    "productivity_0_100": [0, 100],
    "mental_wellness_index_0_100": [0, 100]
})

DEFAULTS = config.get("defaults", {
    "age": 30,
    "sleep_hours": 7.0,
    "screen_time_hours": 8.0,
    "work_screen_hours": 5.0,
    "leisure_screen_hours": 3.0,
    "sleep_quality_1_5": 3,
    "stress_level_0_10": 5,
    "productivity_0_100": 50,
    "exercise_minutes_per_week": 90,
    "social_hours_per_week": 5
})

STATIC_CAT_OPTIONS = {
    "gender": ["Female", "Male", "Non-binary/Other"],
    "occupation": ["Employed", "Student", "Self-employed", "Unemployed", "Retired"],
    "work_mode": ["Remote", "In-person", "Hybrid"]
}


def categorical_options():
    options = {}
    if isinstance(encoder, dict):
        for col in CATEGORICAL:
            le = encoder.get(col)
            if hasattr(le, "classes_"):
                options[col] = list(map(str, le.classes_.tolist()))
            else:
                options[col] = STATIC_CAT_OPTIONS.get(col, [])
    else:
        for col in CATEGORICAL:
            options[col] = STATIC_CAT_OPTIONS.get(col, [])
    return options


def clip_value(col, val):
    if col in CLEANING_RULES:
        lo, hi = CLEANING_RULES[col]
        try:
            v = float(val)
        except Exception:
            return val, None
        if v < lo:
            return lo, f"{col} below min {lo} — clipped to {lo}"
        if v > hi:
            return hi, f"{col} above max {hi} — clipped to {hi}"
        return v, None
    return val, None


def coerce_numeric(x, fallback):
    try:
        return float(x)
    except Exception:
        return float(fallback)


def validate_and_clean_row(raw_row):
    cleaned = {}
    warnings = []
    errors = []
    cat_opts = categorical_options()

    for f in FEATURES:
        val = raw_row.get(f, "")
        if f in NUMERIC:
            if val is None or val == "" or (isinstance(val, str) and val.strip() == ""):
                default = DEFAULTS.get(f, 0.0)
                cleaned_val = coerce_numeric(default, 0.0)
                warnings.append(f"{f} was missing — using default {cleaned_val}")
                clipped, clip_warn = clip_value(f, cleaned_val)
                if clip_warn:
                    warnings.append(clip_warn)
                cleaned[f] = float(clipped)
            else:
                try:
                    num = float(val)
                    clipped, clip_warn = clip_value(f, num)
                    if clip_warn:
                        warnings.append(clip_warn)
                    cleaned[f] = float(clipped)
                except Exception:
                    default = DEFAULTS.get(f, 0.0)
                    cleaned[f] = float(default)
                    warnings.append(f"{f} invalid value -> using default {default}")
        else:
            s = "" if val is None else str(val).strip()
            if f in cat_opts and cat_opts[f]:
                if s == "":
                    default_opt = cat_opts[f][0]
                    cleaned[f] = default_opt
                    warnings.append(f"{f} was empty — defaulting to '{default_opt}'")
                elif s not in cat_opts[f]:
                    warnings.append(f"{f} value '{s}' not in known options; will attempt best-effort encoding")
                    cleaned[f] = s
                else:
                    cleaned[f] = s
            else:
                cleaned[f] = s

    return cleaned, warnings, errors


def require_json(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        return f(*args, **kwargs)
    return decorated


ranges = {
    "age": [10, 90],
    "sleep_hours": [0, 24],
    "sleep_quality_1_5": [1, 5],
    "stress_level_0_10": [0, 10],
    "productivity_0_100": [0, 100],
    "exercise_minutes_per_week": [0, 1000],
    "social_hours_per_week": [0, 168]
}


INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <title>OYASUMI-MIND — Daily Wellness Insights</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style>
        :root {
            --primary: #1976d2;
            --primary-light: #e3f2fd;
            --warn: #f59e0b;
            --warn-bg: #fff7e6;
            --info: #6b7280;
            --info-bg: #f3f4f6;
            --bg: #f9fafb;
            --radius: 10px;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: var(--bg);
            margin: 0;
            padding: 20px;
            color: #222;
        }

        .container {
            max-width: 780px;
            margin: 20px auto;
            background: #fff;
            border-radius: var(--radius);
            padding: 24px 28px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        }

        h1 {
            text-align: center;
            color: var(--primary);
            margin-bottom: 12px;
        }

        p.intro {
            text-align: center;
            margin-top: 0;
            margin-bottom: 24px;
            color: #555;
            line-height: 1.6;
        }

        form div {
            margin-bottom: 18px;
        }

        label {
            font-weight: 600;
            display: block;
            margin-bottom: 8px;
            cursor: pointer;
        }

        input, select {
            width: 100%;
            padding: 10px 12px;
            border: 1px solid #ccc;
            border-radius: var(--radius);
            font-size: 15px;
            transition: border-color 0.2s, box-shadow 0.2s;
            box-sizing: border-box;
        }

        input:focus, select:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px var(--primary-light);
            outline: none;
        }

        select:required:invalid {
            color: #888;
        }

        .btn {
            display: block;
            width: 100%;
            background: var(--primary);
            color: #fff;
            border: none;
            padding: 12px;
            border-radius: var(--radius);
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.25s;
        }

        .btn:hover {
            background: #1565c0;
        }

        .result, .warn, .disclaimer {
            padding: 14px 16px;
            border-left: 5px solid;
            border-radius: var(--radius);
            margin-top: 24px;
            line-height: 1.5;
        }

        .result {
            background: var(--primary-light);
            border-color: var(--primary);
        }

        .warn {
            background: var(--warn-bg);
            border-color: var(--warn);
        }

        .disclaimer {
            background: var(--info-bg);
            border-color: var(--info);
            font-size: 0.9rem;
        }

        pre {
            white-space: pre-wrap;
            word-wrap: break-word;
            font-family: monospace;
            background: #f5f5f5;
            padding: 10px;
            border-radius: var(--radius);
            margin-top: 10px;
            overflow-x: auto;
        }

        details summary {
            cursor: pointer;
            font-size: 0.9em;
            margin-top: 8px;
        }

        footer {
            text-align: center;
            margin-top: 30px;
            font-size: 0.85rem;
            color: #777;
        }

        @media (max-width: 600px) {
            .container {
                padding: 18px;
                margin: 10px auto;
            }
            h1 {
                font-size: 1.4rem;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 OYASUMI-MIND</h1>
        <p class="intro">
            Discover patterns in your daily routine. By answering a few questions, this tool can offer insights into your general wellness based on your habits.
        </p>

        <form method="post" action="/predict" autocomplete="off">
            {% for f in features %}
            <div>
                <label for="{{ f }}">{{ f.replace('_', ' ').title() }}</label>
                
                {% if f in categorical %}
                    <select name="{{ f }}" id="{{ f }}" required>
                        <option value="" disabled selected>Select an option</option>
                        {% for opt in options[f] %}
                        <option value="{{ opt }}">{{ opt }}</option>
                        {% endfor %}
                    </select>

                {% elif f in ranges %}
                    <select name="{{ f }}" id="{{ f }}" required>
                        <option value="" disabled selected>Select {{ ranges[f][0] }}–{{ ranges[f][1] }}</option>
                        {% for i in range(ranges[f][0], ranges[f][1]+1) %}
                        <option value="{{ i }}">{{ i }}</option>
                        {% endfor %}
                    </select>

                {% else %}
                    <input name="{{ f }}" id="{{ f }}" type="number" step="any" required />
                {% endif %}
            </div>
            {% endfor %}

            <button class="btn" type="submit">Analyze My Routine</button>
        </form>

        {% if warnings %}
            {% for w in warnings %}
            <div class="warn">⚠️ <strong>Heads up:</strong> {{ w }}</div>
            {% endfor %}
        {% endif %}

        {% if error %}
        <div class="warn">
            <strong>Oops! Something went wrong.</strong>
            <p>There was a technical issue processing your request. Please try again.</p>
            <pre>{{ error }}</pre>
        </div>
        {% endif %}

        {% if prediction is not none %}
        <div class="result">
            <strong>Your Insight:</strong> {{ prediction }}
            <details>
                <summary>Show my inputs</summary>
                <pre>{{ input_values }}</pre>
            </details>
        </div>
        {% endif %}

        <div class="disclaimer">
            <strong>Important Disclaimer:</strong> This tool is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified health provider with any questions you may have regarding a medical condition.
        </div>
    </div>

    <footer>
        <p>&copy; 2025 OYASUMI-MIND. All rights reserved.</p>
    </footer>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(
        INDEX_HTML,
        features=FEATURES,
        categorical=CATEGORICAL,
        options=categorical_options(),
        prediction=None,
        warnings=None,
        error=None,
        input_values=None,
        ranges=ranges
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        row = {}
        for f in FEATURES:
            raw = request.form.get(f)
            row[f] = raw if raw is not None else ""

        cleaned_row, warnings, errors = validate_and_clean_row(row)

        df = pd.DataFrame([cleaned_row], columns=FEATURES)

        if isinstance(encoder, dict):
            for col in CATEGORICAL:
                le = encoder.get(col)
                if le is None:
                    df[col] = df[col].apply(lambda v: -1 if v == "" or pd.isna(v) else v)
                    continue
                val = df.at[0, col]
                try:
                    encoded = le.transform([val])[0]
                    df[col] = encoded
                except Exception:
                    df[col] = -1
                    warnings.append(f"failed to encode categorical '{col}' -> using -1")
        else:
            if encoder is not None and hasattr(encoder, "transform"):
                try:
                    transformed = encoder.transform(df[CATEGORICAL])
                    if isinstance(transformed, np.ndarray) and transformed.ndim == 2 and transformed.shape[1] == len(CATEGORICAL):
                        for i, col in enumerate(CATEGORICAL):
                            df[col] = transformed[0, i]
                except Exception:
                    logger.debug("encoder.transform failed; continuing")

        for col in NUMERIC:
            if col in df.columns and pd.isna(df.at[0, col]):
                df.at[0, col] = float(DEFAULTS.get(col, 0.0))

        try:
            if model is not None and hasattr(model, "feature_names_in_"):
                needed = list(model.feature_names_in_)
                for c in needed:
                    if c not in df.columns:
                        df[c] = 0
                X = df[needed]
            else:
                X = df
        except Exception:
            X = df

        try:
            X_numeric = X.astype(float)
        except Exception:
            X_numeric = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        if model is None:
            raise CustomException("Model not loaded; set MODEL_PATH or fix load errors. See logs.")

        pred = model.predict(X_numeric)
        pred_value = float(np.asarray(pred).reshape(-1)[0])
        pred_out = round(pred_value, 3)

        return render_template_string(
            INDEX_HTML,
            features=FEATURES,
            categorical=CATEGORICAL,
            options=categorical_options(),
            prediction=pred_out,
            warnings=warnings,
            error=None,
            input_values=json.dumps(cleaned_row, indent=2)
        )


    except CustomException as ce:
        logger.exception("CustomException during prediction: %s", ce)
        return render_template_string(
            INDEX_HTML,
            features=FEATURES,
            prediction=None,
            warnings=None,
            error=str(ce),
            input_values=None
        )
    except Exception as e:
        tb = traceback.format_exc()
        logger.exception("Unexpected error during prediction")
        return render_template_string(
            INDEX_HTML,
            features=FEATURES,
            prediction=None,
            warnings=None,
            error=str(e) + "\n" + tb,
            input_values=None
        )


@app.route("/predict.json", methods=["POST"])
@require_json
def predict_json():
    try:
        payload = request.get_json()
        if not isinstance(payload, dict):
            return jsonify({"error": "JSON body must be an object mapping feature names to values"}), 400

        row = {f: payload.get(f, "") for f in FEATURES}
        cleaned_row, warnings, errors = validate_and_clean_row(row)

        df = pd.DataFrame([cleaned_row], columns=FEATURES)

        if isinstance(encoder, dict):
            for col in CATEGORICAL:
                le = encoder.get(col)
                if le is None:
                    df[col] = df[col].apply(lambda v: -1 if v == "" or pd.isna(v) else v)
                    continue
                val = df.at[0, col]
                try:
                    encoded = le.transform([val])[0]
                    df[col] = encoded
                except Exception:
                    df[col] = -1
                    warnings.append(f"failed to encode categorical '{col}' -> using -1")
        else:
            if encoder is not None and hasattr(encoder, "transform"):
                try:
                    transformed = encoder.transform(df[CATEGORICAL])
                    if isinstance(transformed, np.ndarray) and transformed.ndim == 2 and transformed.shape[1] == len(CATEGORICAL):
                        for i, col in enumerate(CATEGORICAL):
                            df[col] = transformed[0, i]
                except Exception:
                    logger.debug("encoder.transform failed; continuing")

        for col in NUMERIC:
            if col in df.columns and pd.isna(df.at[0, col]):
                df.at[0, col] = float(DEFAULTS.get(col, 0.0))

        try:
            if model is not None and hasattr(model, "feature_names_in_"):
                needed = list(model.feature_names_in_)
                for c in needed:
                    if c not in df.columns:
                        df[c] = 0
                X = df[needed]
            else:
                X = df
        except Exception:
            X = df

        try:
            X_numeric = X.astype(float)
        except Exception:
            X_numeric = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        if model is None:
            return jsonify({"error": "Model not available on server"}), 500

        pred = model.predict(X_numeric)
        pred_value = float(np.asarray(pred).reshape(-1)[0])
        pred_out = round(pred_value, 3)

        return jsonify({
            "prediction": pred_out,
            "cleaned_input": cleaned_row,
            "warnings": warnings,
            "errors": errors
        })

    except CustomException as ce:
        logger.exception("CustomException in predict.json: %s", ce)
        return jsonify({"error": str(ce)}), 500
    except Exception as e:
        logger.exception("Unexpected exception in predict.json")
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}), 200


@app.route("/ready", methods=["GET"])
def ready():
    ready_state = model is not None
    return jsonify({
        "ready": ready_state,
        "model_loaded": ready_state,
        "time": datetime.utcnow().isoformat() + "Z"
    }), (200 if ready_state else 503)


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1" or os.environ.get("FLASK_ENV", "") == "development"
    logger.info("Starting Flask server on %s:%s (debug=%s)", host, port, debug)
    try:
        app.run(host=host, port=port, debug=debug)
    except Exception:
        logger.exception("Exception while running Flask development server")
        raise
