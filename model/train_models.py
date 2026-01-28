import pandas as pd
import numpy as np
import joblib
import os, sys
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, classification_report

# ---------------------------------------------------------
# Import feature list
# ---------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.feature_list import FEATURE_LIST

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
DATA_PATH = "Dataset"
MODEL_PATH = "model"
os.makedirs(MODEL_PATH, exist_ok=True)

CIC_FILE = os.path.join(DATA_PATH, "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")

MODEL_FILE   = os.path.join(MODEL_PATH, "isolation_forest.pkl")
SCALER_FILE  = os.path.join(MODEL_PATH, "scaler.pkl")
ENCODER_FILE = os.path.join(MODEL_PATH, "encoders.pkl")

# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
print("[INFO] Loading CIC-IDS-2017 dataset...")
df = pd.read_csv(CIC_FILE)
print(f"[INFO] Dataset shape: {df.shape}")

# ---------------------------------------------------------
# Clean column names
# ---------------------------------------------------------
df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_").str.replace("-", "_")

column_map = {
   
    "protocol": "protocol_type",
    "flag":"flag",
    "dst_port": "destination_port",
    "flow_duration_ms": "flow_duration",
    "total_fwd_packets": "total_forward_packets",
    "total_backward_packets": "total_backward_packets",
    "average_packet_size": "average_packet_size",
    "flow_bytes_s": "flow_bytes_per_s",
    "fwd_iat_mean": "fwd_iat_mean",
    "bwd_iat_mean": "bwd_iat_mean"
}

df = df.rename(columns={c: column_map.get(c, c) for c in df.columns})

# ---------------------------------------------------------
# Detect label column
# ---------------------------------------------------------
label_col = "label" if "label" in df.columns else None
if label_col:
    print("[INFO] Label column detected")
else:
    print("[WARN] No label column found")


# ---------------------------------------------------------
# Ensure all features exist
# ---------------------------------------------------------
for f in FEATURE_LIST:
    if f not in df.columns:
        df[f] = 0

# ---------------------------------------------------------
# Prepare data
# ---------------------------------------------------------
X = df[FEATURE_LIST]

if label_col:
    y_true = np.where(df[label_col].str.contains("BENIGN", case=False), 1, -1)
else:
    y_true = None

# ---------------------------------------------------------
# Encode categorical features
# ---------------------------------------------------------
encoders = {}

categorical_cols = X.select_dtypes(include=["object"]).columns

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

# Scale (KEEP FEATURE NAMES)
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
    contamination=0.25,
    random_state=42
)
model.fit(X_scaled)
print("[INFO] Training complete")

# ---------------------------------------------------------
# Evaluation (optional but useful)
# ---------------------------------------------------------
if y_true is not None:
    y_pred = model.predict(X_scaled)
    print("\n[INFO] Evaluation:")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=["Attack", "Normal"]))

# ---------------------------------------------------------
# Save artifacts
# ---------------------------------------------------------
joblib.dump(model, MODEL_FILE)
joblib.dump(scaler, SCALER_FILE)
joblib.dump(encoders, ENCODER_FILE)

print("[INFO] Model, scaler, and encoders saved successfully.")
