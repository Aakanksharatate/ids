import pandas as pd
import numpy as np

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.feature_list import FEATURE_LIST
from backend.app import encoders, scaler, model
from backend.signature import check_signature


def predict_intrusion(input_data):
    try:
        # Safety check
        missing = [f for f in FEATURE_LIST if f not in input_data]
        if missing:
            return {"error": f"Missing features: {missing}"}

        # ---------------- SIGNATURE-BASED DETECTION ----------------
        is_attack, reason = check_signature(input_data)
        if is_attack:
            return {
                "prediction": "Attack",
                "confidence": 90.0,
                "method": "Signature-Based",
                "reason": reason
            }
        
        # ------------------------------------------------------------

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

        # 4. ML Prediction (Anomaly-based)
        pred  = model.predict(df_scaled)[0]
        score = model.decision_function(df_scaled)[0]

        label = "Normal" if pred == 1 else "Anomaly"
        confidence = round(float(100 / (1 + np.exp(-score))), 2)

        return {
            "prediction": label,
            "confidence": confidence,
            "method": "Anomaly-Based (ML)"
        }

    except Exception as e:
        return {"error": str(e)}
