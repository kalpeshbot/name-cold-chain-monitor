import pandas as pd
import json
import time
import os
import sys
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import (
    MQTT_BROKER, MQTT_PORT, MQTT_TOPIC,
    TEMP_MIN, TEMP_MAX, HUMIDITY_MIN, HUMIDITY_MAX,
    SHIPMENT_ID, DATA_FILE
)


def subscribe():
    # STEP 1 — Setup
    os.makedirs("logs", exist_ok=True)
    log_path = "logs/mqtt_log.txt"
    with open(log_path, "w") as f:
        pass

    # STEP 2 — Load data
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} rows from {DATA_FILE}")
    print(f"Simulating MQTT stream from broker: {MQTT_BROKER}:{MQTT_PORT}")
    print(f"Topic: {MQTT_TOPIC}")
    print("Starting data stream... (Press Ctrl+C to stop)")

    count = 0

    try:
        # STEP 3 — Process each row
        for _, row in df.iterrows():
            # 3a — Build message dict
            message = {
                "timestamp": str(row["timestamp"]),
                "shipment_id": str(row["shipment_id"]),
                "temperature": round(float(row["temperature"]), 2),
                "humidity": round(float(row["humidity"]), 2),
                "door_open": bool(row["door_open"]),
                "door_open_duration_sec": int(row["door_open_duration_sec"]),
                "latitude": round(float(row["latitude"]), 4),
                "longitude": round(float(row["longitude"]), 4),
                "vehicle_speed_kmh": round(float(row["vehicle_speed_kmh"]), 1),
                "battery_level": round(float(row["battery_level"]), 1),
                "signal_strength": int(row["signal_strength"]),
                "spoilage_probability": round(float(row["spoilage_probability"]), 4),
            }

            # 3b — Check for alerts
            temp_alert = (
                message["temperature"] > TEMP_MAX
                or message["temperature"] < TEMP_MIN
            )
            humidity_alert = (
                message["humidity"] > HUMIDITY_MAX
                or message["humidity"] < HUMIDITY_MIN
            )
            door_alert = message["door_open"] is True

            # 3c — Print to terminal
            timestamp = message["timestamp"]
            temp = message["temperature"]
            humidity = message["humidity"]
            door = message["door_open"]
            speed = message["vehicle_speed_kmh"]
            battery = message["battery_level"]

            print(
                f"[{timestamp}] TEMP: {temp}°C | HUM: {humidity}% | "
                f"DOOR: {door} | SPEED: {speed}km/h | BATT: {battery}%"
            )
            if temp_alert:
                print(
                    f"  ⚠️  TEMPERATURE ALERT: {temp}°C is outside safe range "
                    f"{TEMP_MIN}°C - {TEMP_MAX}°C"
                )
            if humidity_alert:
                print(
                    f"  ⚠️  HUMIDITY ALERT: {humidity}% is outside safe range "
                    f"{HUMIDITY_MIN}% - {HUMIDITY_MAX}%"
                )
            if door_alert:
                print(
                    f"  🚪 DOOR OPEN EVENT detected — duration: "
                    f"{message['door_open_duration_sec']} seconds"
                )

            # 3d — Append log entry
            log_line = (
                f"[{timestamp}] {message['shipment_id']} | "
                f"TEMP: {temp:.2f}°C | HUM: {humidity:.2f}% | "
                f"DOOR: {door} | LAT: {message['latitude']} | "
                f"LON: {message['longitude']} | "
                f"SPOILAGE: {message['spoilage_probability'] * 100:.2f}%\n"
            )
            with open(log_path, "a") as f:
                f.write(log_line)

            # 3e — Sleep
            time.sleep(0.05)
            count += 1

        # STEP 4 — Completion
        print(f"Stream complete. Total messages processed: {count}")
        print("Log saved to logs/mqtt_log.txt")

    except KeyboardInterrupt:
        # STEP 5 — Error handling
        print("Stream interrupted by user. Partial log saved.")


if __name__ == "__main__":
    subscribe()
