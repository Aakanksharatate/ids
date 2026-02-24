import pandas as pd
import numpy as np
import joblib
import os, sys
import gc

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, classification_report

# ---------------------------------------------------------
# Import feature list
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from utils.feature_list import FEATURE_LIST

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
DATA_PATH  = os.path.join(BASE_DIR, "Dataset")
MODEL_PATH = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_PATH, exist_ok=True)

FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
]

MODEL_FILE   = os.path.join(MODEL_PATH, "isolation_forest.pkl")
SCALER_FILE  = os.path.join(MODEL_PATH, "scaler.pkl")
ENCODER_FILE = os.path.join(MODEL_PATH, "encoders.pkl")

# ---------------------------------------------------------
# Load datasets
# ---------------------------------------------------------
dfs = []
for file in FILES:
    path = os.path.join(DATA_PATH, file)
    print(f"[INFO] Loading {file} ...")
    dfs.append(pd.read_csv(path))

df = pd.concat(dfs, ignore_index=True)
print(f"[INFO] Combined dataset shape: {df.shape}")

gc.collect()

# ---------------------------------------------------------
# Clean column names
# ---------------------------------------------------------
df.columns = (
    df.columns
    .str.lower()
    .str.strip()
    .str.replace(" ", "_")
    .str.replace("-", "_")
)

# ---------------------------------------------------------
# Column mapping (COMPLETE)
# ---------------------------------------------------------
column_map = {
    "protocol": "protocol_type",
    "flag": "flag",
    "dst_port": "destination_port",
    "flow_duration": "flow_duration",
    "total_fwd_packets": "total_forward_packets",
    "total_backward_packets": "total_backward_packets",
    "average_packet_size": "average_packet_size",
    "flow_bytes_s": "flow_bytes_per_s",
    "fwd_iat_mean": "fwd_iat_mean",
    "bwd_iat_mean": "bwd_iat_mean"
}

# Create mapped columns WITHOUT full DataFrame rename
for old, new in column_map.items():
    if old in df.columns:
        df[new] = df[old]

# ---------------------------------------------------------
# Detect label column
# ---------------------------------------------------------
label_col = "label" if "label" in df.columns else None
print("[INFO] Label column detected" if label_col else "[WARN] No label column found")

# ---------------------------------------------------------
# Ensure all features exist (fill missing with 0)
# ---------------------------------------------------------
missing_features = [f for f in FEATURE_LIST if f not in df.columns]
if missing_features:
    print("[INFO] Missing features filled with 0:", missing_features)
    for f in missing_features:
        df[f] = 0

# ---------------------------------------------------------
# Keep only required columns (EARLY FEATURE SELECTION)
# ---------------------------------------------------------
final_cols = FEATURE_LIST + ([label_col] if label_col else [])
df = df[final_cols]

gc.collect()

# ---------------------------------------------------------
# Prepare data
# ---------------------------------------------------------
X = df[FEATURE_LIST]

if label_col:
    y_true = np.where(
        df[label_col].astype(str).str.contains("BENIGN", case=False),
        1,
        -1
    )
else:
    y_true = None

# ---------------------------------------------------------
# Encode categorical features
# ---------------------------------------------------------
encoders = {}
for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

# ---------------------------------------------------------
# Scale features (KEEP FEATURE NAMES)
# ---------------------------------------------------------
scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=FEATURE_LIST
)

# ---------------------------------------------------------
# Train Isolation Forest
# ---------------------------------------------------------
print("[INFO] Training Isolation Forest...")
model = IsolationForest(
    n_estimators=300,
    contamination=0.20,
    random_state=42,
    n_jobs=-1
)
model.fit(X_scaled)
print("[INFO] Training complete")

# ---------------------------------------------------------
# Evaluation (INFORMATIONAL ONLY)
# ---------------------------------------------------------
if y_true is not None:
    y_pred = model.predict(X_scaled)
    print("\n[INFO] Evaluation:")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, zero_division=0))

# ---------------------------------------------------------
# Save artifacts
# ---------------------------------------------------------
joblib.dump(model, MODEL_FILE)
joblib.dump(scaler, SCALER_FILE)
joblib.dump(encoders, ENCODER_FILE)

print("[INFO] Model, scaler, and encoders saved successfully.")
