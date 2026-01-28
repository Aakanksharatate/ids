# backend/predict.py

import pandas as pd
import numpy as np
import joblib
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.feature_list import FEATURE_LIST

# ---------------------------------------------------------
# Load trained artifacts ONLY
# ---------------------------------------------------------
MODEL_FILE   = "model/isolation_forest.pkl"
SCALER_FILE  = "model/scaler.pkl"
ENCODER_FILE = "model/encoders.pkl"

model    = joblib.load(MODEL_FILE)
scaler   = joblib.load(SCALER_FILE)
encoders = joblib.load(ENCODER_FILE)

def predict_intrusion(input_data):
    try:
        # Safety check
        missing = [f for f in FEATURE_LIST if f not in input_data]
        if missing:
            return {"error": f"Missing features: {missing}"}

        # 1. Create DataFrame in correct order
        df = pd.DataFrame(
            [[input_data[f] for f in FEATURE_LIST]],
            columns=FEATURE_LIST
        )

        # 2. Encode categorical features
        for col, le in encoders.items():
            df[col] = le.transform(df[col].astype(str))

        # 3. Scale using trained scaler
        df_scaled = pd.DataFrame(
            scaler.transform(df),
            columns=FEATURE_LIST
        )

        # 4. Predict
        pred  = model.predict(df_scaled)[0]
        score = model.decision_function(df_scaled)[0]

        label = "Normal" if pred == 1 else "Anomaly"
        confidence = round(float(100 / (1 + np.exp(-score))), 2)

        return {
            "prediction": label,
            "confidence": confidence
        }

    except Exception as e:
        return {"error": str(e)}

# ---------------------------------------------------------
# Standalone test
# ---------------------------------------------------------
if __name__ == "__main__":
    sample_input = {
        "protocol_type": 1,
        "flag": 2,
        "destination_port": 80,
        "flow_duration": 500,
        "total_forward_packets": 10,
        "total_backward_packets": 8,
        "average_packet_size": 250,
        "flow_bytes_per_s": 1500,
        "fwd_iat_mean": 200,
        "bwd_iat_mean": 180
    }

    print(predict_intrusion(sample_input))
