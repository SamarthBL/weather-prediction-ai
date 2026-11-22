"""Flask API and website for the Weather Prediction AI mini project."""
import json
import os
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from train_model import FEATURES, MODEL_DIR, train

ROOT = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(ROOT / "static"))


def ensure_models():
    model_path = MODEL_DIR / "weather_models.joblib"
    metrics_path = MODEL_DIR / "metrics.json"
    if not model_path.exists() or not metrics_path.exists():
        train()
    return joblib.load(model_path)


def vectorize(readings):
    fields = ["temp_max", "temp_min", "humidity", "pressure", "wind_speed", "cloud_cover", "rainfall"]
    values = {key: float(readings[key]) for key in fields}
    day_of_year = int(readings.get("day_of_year", 180))
    if not 1 <= day_of_year <= 366:
        raise ValueError("day_of_year must be between 1 and 366")
    values["temp_range"] = values["temp_max"] - values["temp_min"]
    values["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    values["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    return [[values[feature] for feature in FEATURES]]


@app.get("/")
def home():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/healthz")
def health():
    return jsonify({"status": "ok"})


@app.get("/api/metrics")
def metrics():
    metrics_path = MODEL_DIR / "metrics.json"
    if not metrics_path.exists():
        train()
    return jsonify(json.loads(metrics_path.read_text(encoding="utf-8")))


@app.post("/api/predict")
def predict():
    readings = request.get_json(silent=True)
    if not isinstance(readings, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400
    try:
        model_bundle = ensure_models()
        features = vectorize(readings)
        temperature = float(model_bundle["regressor"].predict(features)[0])
        rain_probability = float(model_bundle["classifier"].predict_proba(features)[0, 1])
        return jsonify({
            "temperature_c": round(temperature, 1),
            "rain_probability": round(rain_probability, 4),
            "model": "Random Forest",
            "mode": "trained-model",
        })
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    train()
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
