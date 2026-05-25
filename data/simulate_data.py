import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

N_ROWS = 2880
SHIPMENT_ID = "SHIPMENT_1234"
START_TIME = datetime(2024, 1, 15, 8, 0, 0)
DOOR_EVENT_ROWS = [300, 600, 850, 950, 1100, 1400, 1800, 2400]


def _temperature_series(rng):
    temps = np.zeros(N_ROWS)

    # Phase 1 (rows 0-500): random walk 3.5-6.5
    temps[0] = 5.0
    for i in range(1, 501):
        temps[i] = np.clip(temps[i - 1] + rng.normal(0, 0.15), 3.5, 6.5)

    # Phase 2 (rows 501-900): upward drift 4.0-7.2
    base = np.linspace(4.0, 7.2, 400)
    walk = np.cumsum(rng.normal(0, 0.08, 400))
    temps[501:901] = np.clip(base + walk - walk[0], 4.0, 7.2)

    # Phase 3 (rows 901-1200): anomaly rise 7.2 to 9.5
    temps[901:1201] = np.linspace(7.2, 9.5, 300)

    # Phase 4 (rows 1201-1500): recovery 9.5 down toward 5.0-7.0
    temps[1201:1501] = np.linspace(temps[1200], 5.5, 300)

    # Phase 5 (rows 1501-2880): normal random walk 3.8-6.8
    temps[1501] = 5.5
    for i in range(1502, N_ROWS):
        temps[i] = np.clip(temps[i - 1] + rng.normal(0, 0.12), 3.8, 6.8)

    temps += rng.normal(0, 0.1, N_ROWS)
    return temps


def _humidity_series(rng, door_rows):
    humidity = rng.uniform(48, 58, N_ROWS)
    humidity += rng.normal(0, 0.5, N_ROWS)

    for event_row in door_rows:
        spike = rng.uniform(5, 8)
        humidity[event_row] += spike
        for offset in range(1, 21):
            decay_row = event_row + offset
            if decay_row < N_ROWS:
                factor = 1.0 - offset / 20.0
                humidity[decay_row] += spike * factor

    return humidity


def _spoilage_series(rng):
    spoilage = np.zeros(N_ROWS)
    spoilage[0:901] = np.linspace(0.01, 0.15, 901)
    spoilage[901:1201] = np.linspace(0.15, 0.92, 300)
    spoilage[1201:1501] = np.linspace(0.92, 0.45, 300)
    spoilage[1501:N_ROWS] = np.linspace(0.45, 0.30, N_ROWS - 1501)

    spoilage += rng.normal(0, 0.01, N_ROWS)
    return np.clip(spoilage, 0.0, 1.0)


def _compliance_series(temperature, door_open):
    score = 99.0
    compliance = np.zeros(N_ROWS)
    for i in range(N_ROWS):
        if temperature[i] < 2.0 or temperature[i] > 8.0:
            score -= 0.01
        if door_open[i]:
            score -= 0.5
        score = max(score, 0.0)
        compliance[i] = round(score, 2)
    return compliance


def simulate():
    rng = np.random.default_rng(42)

    door_open = np.zeros(N_ROWS, dtype=bool)
    door_open_duration_sec = np.zeros(N_ROWS, dtype=int)
    duration_rng = np.random.default_rng(42)
    for row in DOOR_EVENT_ROWS:
        door_open[row] = True
        door_open_duration_sec[row] = duration_rng.integers(15, 46)

    temperature = _temperature_series(rng)
    humidity = _humidity_series(rng, DOOR_EVENT_ROWS)

    timestamps = [
        START_TIME + timedelta(seconds=30 * i) for i in range(N_ROWS)
    ]

    latitude = 13.0827 + np.arange(N_ROWS) * 0.0004
    longitude = 80.2707 + np.arange(N_ROWS) * 0.0004

    vehicle_speed_kmh = np.zeros(N_ROWS)
    for i in range(N_ROWS):
        if door_open[i]:
            vehicle_speed_kmh[i] = 0.0
        else:
            vehicle_speed_kmh[i] = rng.uniform(30, 60)

    battery_level = np.round(100 - 0.035 * np.arange(N_ROWS), 1)
    signal_strength = rng.integers(72, 96, N_ROWS)
    spoilage_probability = _spoilage_series(rng)
    compliance_score = _compliance_series(temperature, door_open)

    df = pd.DataFrame(
        {
            "timestamp": [t.strftime("%Y-%m-%d %H:%M:%S") for t in timestamps],
            "shipment_id": SHIPMENT_ID,
            "temperature": np.round(temperature, 1),
            "humidity": np.round(humidity, 1),
            "door_open": door_open,
            "door_open_duration_sec": door_open_duration_sec.astype(int),
            "latitude": np.round(latitude, 4),
            "longitude": np.round(longitude, 4),
            "vehicle_speed_kmh": np.round(vehicle_speed_kmh, 1),
            "battery_level": battery_level,
            "signal_strength": signal_strength,
            "spoilage_probability": np.round(spoilage_probability, 2),
            "compliance_score": compliance_score,
        }
    )

    column_order = [
        "timestamp",
        "shipment_id",
        "temperature",
        "humidity",
        "door_open",
        "door_open_duration_sec",
        "latitude",
        "longitude",
        "vehicle_speed_kmh",
        "battery_level",
        "signal_strength",
        "spoilage_probability",
        "compliance_score",
    ]
    return df[column_order]


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    df = simulate()
    df.to_csv("data/sample_data.csv", index=False)
    print(f"Generated {len(df)} rows -> data/sample_data.csv")
    print(f"Temperature range: {df['temperature'].min():.2f} to {df['temperature'].max():.2f}")
    print(f"Door events: {df['door_open'].sum()}")
    print(f"Final compliance score: {df['compliance_score'].iloc[-1]}")
