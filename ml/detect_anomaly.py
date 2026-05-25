import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import DATA_FILE, MODEL_PATH, ANOMALY_THRESHOLD, TEMP_MIN, TEMP_MAX


def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i : i + seq_length])
    return np.array(sequences)


def detect():
    # STEP 1 — Load artifacts
    model = keras.models.load_model(MODEL_PATH)
    with open("ml/model/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("ml/model/threshold.pkl", "rb") as f:
        threshold = pickle.load(f)
    print(f"Model loaded. Anomaly threshold: {threshold:.6f}")

    # STEP 2 — Load and scale full dataset
    df = pd.read_csv(DATA_FILE)
    features = df[["temperature", "humidity", "spoilage_probability"]].values
    scaled_data = scaler.transform(features)

    # STEP 3 — Create sequences
    seq_length = 20
    X = create_sequences(scaled_data, seq_length)

    # STEP 4 — Predict and calculate reconstruction error
    predictions = model.predict(X)
    errors = np.mean(np.abs(X - predictions), axis=(1, 2))
    is_anomaly_seq = errors > threshold
    ml_spoilage_seq = np.clip(errors / (threshold * 2), 0.0, 1.0)
    ml_spoilage_seq = np.round(ml_spoilage_seq, 4)

    # STEP 5 — Build result DataFrame
    n_rows = len(df)
    reconstruction_error = np.full(n_rows, np.nan)
    is_anomaly = np.full(n_rows, np.nan)
    ml_spoilage_probability = np.full(n_rows, np.nan)

    for i, err in enumerate(errors):
        row_idx = i + seq_length - 1
        reconstruction_error[row_idx] = round(float(err), 6)
        is_anomaly[row_idx] = bool(is_anomaly_seq[i])
        ml_spoilage_probability[row_idx] = ml_spoilage_seq[i]

    df["reconstruction_error"] = reconstruction_error
    df["is_anomaly"] = is_anomaly
    df["ml_spoilage_probability"] = ml_spoilage_probability
    df["alert_triggered"] = (df["is_anomaly"] == True) & (df["temperature"] > TEMP_MAX)

    # STEP 6 — Print summary
    total = len(df)
    anomaly_count = int(df["is_anomaly"].sum())
    alert_count = int(df["alert_triggered"].sum())
    peak_error = float(df["reconstruction_error"].max())
    print(f"Total rows processed: {total}")
    print(f"Anomalies detected: {anomaly_count}")
    print(f"Alerts triggered (temp exceeded): {alert_count}")
    print(f"Peak reconstruction error: {peak_error:.6f}")
    print("Detection complete.")

    # STEP 7 — Save results
    df.to_csv("data/anomaly_results.csv", index=False)
    print("Results saved to data/anomaly_results.csv")

    # STEP 8 — Return
    return df


if __name__ == "__main__":
    df_results = detect()
    print(df_results[["timestamp", "temperature", "is_anomaly", "alert_triggered"]].tail(20))
