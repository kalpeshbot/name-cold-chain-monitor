import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import DATA_FILE, MODEL_PATH, TEMP_MIN, TEMP_MAX


def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i : i + seq_length])
    return np.array(sequences)


def train():
    # STEP 1 — Load data
    df = pd.read_csv(DATA_FILE)
    features = df[["temperature", "humidity", "spoilage_probability"]]
    df_normal = df[(df["temperature"] >= TEMP_MIN) & (df["temperature"] <= TEMP_MAX)]
    normal_features = df_normal[["temperature", "humidity", "spoilage_probability"]].values
    print(f"Training samples (normal only): {len(df_normal)}")

    # STEP 2 — Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(normal_features)
    os.makedirs("ml/model", exist_ok=True)
    with open("ml/model/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("Scaler saved to ml/model/scaler.pkl")

    # STEP 3 — Create sequences
    seq_length = 20
    X = create_sequences(scaled_data, seq_length)
    print(f"Sequences created: {X.shape}")

    # STEP 4 — Build LSTM Autoencoder
    inputs = keras.Input(shape=(20, 3))
    encoded = layers.LSTM(64, activation="relu", return_sequences=True)(inputs)
    encoded = layers.LSTM(32, activation="relu", return_sequences=False)(encoded)
    encoded = layers.RepeatVector(20)(encoded)
    decoded = layers.LSTM(32, activation="relu", return_sequences=True)(encoded)
    decoded = layers.LSTM(64, activation="relu", return_sequences=True)(decoded)
    outputs = layers.TimeDistributed(layers.Dense(3))(decoded)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mae",
    )
    model.summary()

    # STEP 5 — Train
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )
    history = model.fit(
        X,
        X,
        epochs=30,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1,
    )
    with open("ml/model/training_history.pkl", "wb") as f:
        pickle.dump(history.history, f)

    # STEP 6 — Calculate threshold
    predictions = model.predict(X)
    errors = np.mean(np.abs(X - predictions), axis=(1, 2))
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    threshold = mean_error + (2 * std_error)
    with open("ml/model/threshold.pkl", "wb") as f:
        pickle.dump(threshold, f)
    print(f"Anomaly threshold set at: {threshold:.6f}")

    # STEP 7 — Save model
    model.save(MODEL_PATH)
    print("Model saved to ml/model/lstm_model.keras")
    print("Training complete.")


if __name__ == "__main__":
    train()
