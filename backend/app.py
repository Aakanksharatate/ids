from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.feature_list import FEATURE_LIST
from backend.signature import check_signature


# Initialize Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")

# Paths for models
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../model/isolation_forest.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "../model/scaler.pkl")
ENCODERS_PATH = os.path.join(os.path.dirname(__file__), "../model/encoders.pkl")

# Load model components
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoders = joblib.load(ENCODERS_PATH)

# Import feature list
from utils.feature_list import FEATURE_LIST


@app.route("/")
def home():
    """Render the main dashboard page"""
    return render_template("dashboard.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Handle predictions from the frontend"""
    try:
        data = request.get_json()
        df = pd.DataFrame([data])

        # Encode categorical features
        for col, le in encoders.items():
            if col in df.columns:
                df[col] = le.transform(df[col])

        # Scale numerical features
        X_scaled = scaler.transform(df[FEATURE_LIST])

        # Predict using Isolation Forest
        pred = model.predict(X_scaled)[0]
        score = model.decision_function(X_scaled)[0]
        confidence = (1 - np.exp(-abs(score))) * 100

        result = "Normal" if pred == 1 else "Attack"

        return jsonify({
            "prediction": result,
            "confidence": round(float(confidence), 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True, port=5000)