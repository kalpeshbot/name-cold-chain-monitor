import pandas as pd
import requests
import json
import time
import os
import sys
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import (
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    TEMP_MIN,
    TEMP_MAX,
    SHIPMENT_ID,
    DRIVER_NAME,
    VEHICLE_NUMBER,
)


def _group_alert_events(alert_rows, df):
    """Group consecutive alert rows with gap > 5 rows between groups."""
    if alert_rows.empty:
        return []

    alert_indices = alert_rows.index.tolist()
    groups = []
    current_group = [alert_indices[0]]

    for i in range(1, len(alert_indices)):
        if alert_indices[i] - alert_indices[i - 1] > 5:
            groups.append(current_group)
            current_group = [alert_indices[i]]
        else:
            current_group.append(alert_indices[i])
    groups.append(current_group)

    events = []
    for group_indices in groups:
        group_df = df.loc[group_indices]
        peak_temp_idx = group_df["temperature"].idxmax()
        peak_row = df.loc[peak_temp_idx]

        events.append(
            {
                "event_start": str(group_df.iloc[0]["timestamp"]),
                "event_end": str(group_df.iloc[-1]["timestamp"]),
                "peak_temp": float(group_df["temperature"].max()),
                "peak_spoilage": float(group_df["ml_spoilage_probability"].max() * 100),
                "peak_error": float(group_df["reconstruction_error"].max()),
                "location_lat": float(peak_row["latitude"]),
                "location_lon": float(peak_row["longitude"]),
                "duration_minutes": len(group_df) * 0.5,
                "rows_in_event": len(group_df),
            }
        )

    return events


def send_alerts():
    # STEP 1 — Setup
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    os.makedirs("logs", exist_ok=True)
    log_path = "logs/alert_log.txt"

    # STEP 2 — Load anomaly results
    results_path = "data/anomaly_results.csv"
    if not os.path.exists(results_path):
        print("ERROR: data/anomaly_results.csv not found. Run ml/detect_anomaly.py first.")
        return

    df = pd.read_csv(results_path)
    alert_rows = df[df["alert_triggered"] == True]
    print(f"Found {len(alert_rows)} alert events to send.")

    if len(alert_rows) == 0:
        print("No alerts to send. All readings within safe range.")
        return

    # STEP 3 — Group consecutive alerts into events
    events = _group_alert_events(alert_rows, df)
    total_events = len(events)
    sent_count = 0
    failed_count = 0

    # STEP 4–6 — Send Telegram message for each event group
    for i, event in enumerate(events):
        message = f"""🚨 COLD CHAIN ALERT 🚨

Shipment ID  : {SHIPMENT_ID}
Driver       : {DRIVER_NAME}
Vehicle      : {VEHICLE_NUMBER}

⏰ Event Start : {event['event_start']}
⏰ Event End   : {event['event_end']}
⏱ Duration    : {event['duration_minutes']:.1f} minutes

🌡 Peak Temp   : {event['peak_temp']:.2f}°C  (Safe max: {TEMP_MAX}°C)
☣️ Spoilage Risk: {event['peak_spoilage']:.1f}%
📊 Anomaly Score: {event['peak_error']:.6f}

📍 Location:
   Lat: {event['location_lat']:.4f}
   Lon: {event['location_lon']:.4f}
   Maps: https://maps.google.com/?q={event['location_lat']:.4f},{event['location_lon']:.4f}

⚠️ RECOMMENDED ACTION:
   Spoilage risk > 80%  → Reject shipment immediately
   Spoilage risk 50-80% → Inspect and quarantine
   Spoilage risk < 50%  → Monitor closely and document

🔗 Full audit report will be generated automatically."""

        try:
            response = requests.post(
                url,
                json={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": message,
                    "parse_mode": "HTML",
                },
                timeout=10,
            )

            # STEP 5 — Handle Telegram API response
            if response.status_code == 200:
                print(f"✅ Alert {i + 1}/{total_events} sent successfully.")
                with open(log_path, "a") as f:
                    f.write(
                        f"[{datetime.now()}] ALERT SENT | Event: {event['event_start']} "
                        f"to {event['event_end']} | Peak: {event['peak_temp']:.2f}°C | "
                        f"Spoilage: {event['peak_spoilage']:.1f}%\n"
                    )
                sent_count += 1
            else:
                print(f"❌ Failed to send alert {i + 1}. Status: {response.status_code}")
                print(f"Response: {response.text}")
                with open(log_path, "a") as f:
                    f.write(
                        f"[{datetime.now()}] ALERT FAILED | Status: {response.status_code}\n"
                    )
                failed_count += 1

        except requests.exceptions.RequestException as e:
            # STEP 8 — Error handling
            print(f"❌ Network error sending alert {i + 1}: {str(e)}")
            failed_count += 1

        # STEP 6 — Rate limiting
        if i < total_events - 1:
            time.sleep(1)

    # STEP 7 — Completion summary
    print("Alert sending complete.")
    print(f"Total events: {total_events}")
    print(f"Successfully sent: {sent_count}")
    print(f"Failed: {failed_count}")
    print("Alert log saved to logs/alert_log.txt")


if __name__ == "__main__":
    send_alerts()
