import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import os
import sys
import json
import base64
import io
import hashlib
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from config import (
    TEMP_MIN, TEMP_MAX, HUMIDITY_MIN, HUMIDITY_MAX,
    SHIPMENT_ID, DRIVER_NAME, VEHICLE_NUMBER
)

USERS = {
    "admin": {
        "password": "coldchain123",
        "name": "Admin User",
        "role": "Logistics Manager",
        "avatar": "AC"
    },
    "driver": {
        "password": "driver123",
        "name": "Rajesh Kumar",
        "role": "Driver",
        "avatar": "RK"
    }
}

SESSIONS = {}


def create_session(username):
    token = hashlib.sha256(f"{username}{time.time()}".encode()).hexdigest()[:32]
    SESSIONS[token] = {"username": username, "created": time.time()}
    return token


def get_session(token):
    if token in SESSIONS:
        return SESSIONS[token]
    return None


def delete_session(token):
    if token in SESSIONS:
        del SESSIONS[token]


def get_cookie_token(handler):
    cookie_header = handler.headers.get("Cookie", "")
    for part in cookie_header.split(";"):
        part = part.strip()
        if part.startswith("session_token="):
            return part.split("=", 1)[1]
    return None


def load_data():
    anomaly_path = "data/anomaly_results.csv"
    sample_path = "data/sample_data.csv"

    if os.path.exists(anomaly_path):
        df = pd.read_csv(anomaly_path)
    elif os.path.exists(sample_path):
        df = pd.read_csv(sample_path)
    else:
        rng = np.random.default_rng(42)
        n = 2880
        temperature = np.clip(rng.normal(4.5, 1.5, n), 1, 10)
        row_index = np.arange(n)
        timestamps = pd.date_range("2024-01-15 08:00:00", periods=n, freq="30s")
        df = pd.DataFrame({
            "timestamp": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
            "temperature": temperature,
            "humidity": rng.uniform(48, 58, n),
            "door_open": rng.choice([True, False], n, p=[0.05, 0.95]),
            "door_open_duration_sec": 0,
            "latitude": 13.0827 + row_index * 0.0004,
            "longitude": 80.2707 + row_index * 0.0004,
            "vehicle_speed_kmh": rng.uniform(30, 60, n),
            "battery_level": np.round(100 - row_index * 0.035, 1),
            "signal_strength": rng.integers(72, 96, n),
            "spoilage_probability": np.clip((temperature - 2) / 10, 0, 1),
            "compliance_score": 99.0 - (row_index * 0.01),
            "is_anomaly": temperature > 8.0,
            "alert_triggered": temperature > 8.0,
            "ml_spoilage_probability": np.clip((temperature - 2) / 10, 0, 1),
            "reconstruction_error": np.abs(temperature - 5.0) / 10,
        })

    if "is_anomaly" not in df.columns:
        df["is_anomaly"] = df["temperature"] > TEMP_MAX
    if "alert_triggered" not in df.columns:
        df["alert_triggered"] = df["is_anomaly"]
    if "ml_spoilage_probability" not in df.columns:
        df["ml_spoilage_probability"] = df["spoilage_probability"]
    if "reconstruction_error" not in df.columns:
        df["reconstruction_error"] = 0.0

    return df


def get_kpis(df):
    ts0 = pd.to_datetime(df["timestamp"].iloc[0])
    ts1 = pd.to_datetime(df["timestamp"].iloc[-1])
    return {
        "avg_temp": round(df["temperature"].mean(), 2),
        "max_temp": round(df["temperature"].max(), 2),
        "min_temp": round(df["temperature"].min(), 2),
        "avg_humidity": round(df["humidity"].mean(), 2),
        "total_alerts": int(df["alert_triggered"].sum()),
        "total_door_events": int(df["door_open"].sum()),
        "peak_spoilage": round(df["ml_spoilage_probability"].max() * 100, 1),
        "final_compliance": round(df["compliance_score"].iloc[-1], 2),
        "compliance_status": "PASS" if df["compliance_score"].iloc[-1] >= 85.0 else "FAIL",
        "total_rows": len(df),
        "anomaly_count": int(df["is_anomaly"].sum()),
        "shipment_id": SHIPMENT_ID,
        "driver": DRIVER_NAME,
        "vehicle": VEHICLE_NUMBER,
        "delivery_start": str(df["timestamp"].iloc[0]),
        "delivery_end": str(df["timestamp"].iloc[-1]),
        "avg_speed": round(df["vehicle_speed_kmh"].mean(), 1) if "vehicle_speed_kmh" in df.columns else 0,
        "battery_level": round(df["battery_level"].iloc[-1], 1) if "battery_level" in df.columns else 100,
        "signal_strength": int(df["signal_strength"].iloc[-1]) if "signal_strength" in df.columns else 90,
        "excursion_minutes": round(len(df[df["temperature"] > TEMP_MAX]) * 0.5, 1),
        "total_duration_hours": round((ts1 - ts0).seconds / 3600, 2),
    }


def _fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _style_chart_ax(ax, fig=None):
    ax.set_facecolor("#1a1d2e")
    ax.tick_params(colors="#8b8fa8")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("#e0e0e0")
    ax.grid(color="#2a2d3e", linestyle="--", alpha=0.5)
    if fig is not None:
        for cbar_ax in fig.axes:
            if cbar_ax != ax:
                cbar_ax.tick_params(colors="white")
                cbar_ax.yaxis.label.set_color("white")
                plt.setp(cbar_ax.yaxis.get_ticklabels(), color="white")


def generate_temp_chart(df):
    fig, ax = plt.subplots(figsize=(11, 3.8), facecolor="#0f1117")
    row_index = np.arange(len(df))
    _style_chart_ax(ax)
    ax.axhspan(TEMP_MIN, TEMP_MAX, alpha=0.12, color="#00d4aa")
    ax.fill_between(row_index, df["temperature"], TEMP_MIN, alpha=0.08, color="#4fc3f7")
    ax.plot(row_index, df["temperature"], color="#4fc3f7", linewidth=1, alpha=0.9)
    anomaly_rows = df[df["is_anomaly"] == True]
    if not anomaly_rows.empty:
        ax.scatter(anomaly_rows.index, anomaly_rows["temperature"], color="#ff5252", s=10, zorder=5)
    ax.axhline(TEMP_MIN, color="#ffb74d", linestyle="--", linewidth=1)
    ax.axhline(TEMP_MAX, color="#ff5252", linestyle="--", linewidth=1)
    ax.set_title("Temperature Timeline", color="#e0e0e0", fontsize=11)
    ax.set_ylabel("°C")
    ax.set_xlabel("Reading Index")
    ax.text(0.98, 0.95, f"Avg: {df['temperature'].mean():.1f}°C", transform=ax.transAxes,
            color="#4fc3f7", fontsize=9, ha="right", va="top")
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_spoilage_chart(df):
    fig, ax = plt.subplots(figsize=(11, 3.8), facecolor="#0f1117")
    row_index = np.arange(len(df))
    spoilage = df["ml_spoilage_probability"].fillna(0)
    _style_chart_ax(ax)
    ax.fill_between(row_index, spoilage, 0, color="#ef5350", alpha=0.25)
    ax.plot(row_index, spoilage, color="#ff7043", linewidth=1.2)
    ax.axhline(0.5, color="#ffb74d", linestyle="--", linewidth=1, label="Warning")
    ax.axhline(0.8, color="#ef5350", linestyle="--", linewidth=1, label="Critical")
    ax.axhspan(0.8, 1.05, alpha=0.08, color="#ef5350")
    ax.set_title("Spoilage Risk Index", color="#e0e0e0", fontsize=11)
    ax.set_ylabel("Risk Score")
    ax.set_xlabel("Reading Index")
    ax.legend(fontsize=8, facecolor="#1a1d2e", edgecolor="#2a2d3e", labelcolor="white")
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_humidity_chart(df):
    fig, ax = plt.subplots(figsize=(11, 3.8), facecolor="#0f1117")
    row_index = np.arange(len(df))
    _style_chart_ax(ax)
    ax.axhspan(HUMIDITY_MIN, HUMIDITY_MAX, alpha=0.12, color="#29b6f6")
    ax.fill_between(row_index, df["humidity"], HUMIDITY_MIN, alpha=0.08, color="#29b6f6")
    ax.plot(row_index, df["humidity"], color="#29b6f6", linewidth=1)
    door_rows = df[df["door_open"] == True]
    if not door_rows.empty:
        ax.scatter(door_rows.index, door_rows["humidity"], color="#ffd54f", marker="^", s=40, zorder=5, label="Door Open")
    ax.set_title("Humidity Monitoring", color="#e0e0e0", fontsize=11)
    ax.set_ylabel("%")
    ax.set_xlabel("Reading Index")
    ax.legend(fontsize=8, facecolor="#1a1d2e", edgecolor="#2a2d3e", labelcolor="white")
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_gps_chart(df):
    fig, ax = plt.subplots(figsize=(11, 3.8), facecolor="#0f1117")
    _style_chart_ax(ax, fig)
    ax.plot(df["longitude"], df["latitude"], color="#3a3d5e", linewidth=1.5, alpha=0.6, zorder=1)
    sc = ax.scatter(df["longitude"], df["latitude"], c=df["temperature"], cmap="RdYlGn_r",
                    vmin=TEMP_MIN, vmax=TEMP_MAX + 2, s=6, alpha=0.8)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.85, label="Temp (°C)")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    cbar.set_label("Temp (°C)", color="white")
    ax.scatter(df["longitude"].iloc[0], df["latitude"].iloc[0], color="#69f0ae", marker="^", s=120, zorder=6, label="Start")
    ax.scatter(df["longitude"].iloc[-1], df["latitude"].iloc[-1], color="#ff5252", marker="v", s=120, zorder=6, label="End")
    alert_rows = df[df["alert_triggered"] == True]
    if not alert_rows.empty:
        ax.scatter(alert_rows["longitude"], alert_rows["latitude"], color="#ff1744", marker="x", s=50, zorder=7, label="Alert")
    ax.set_title("GPS Route Map", color="#e0e0e0", fontsize=11)
    ax.legend(fontsize=8, facecolor="#1a1d2e", edgecolor="#2a2d3e", labelcolor="white")
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_reconstruction_chart(df):
    fig, ax = plt.subplots(figsize=(11, 3.8), facecolor="#0f1117")
    row_index = np.arange(len(df))
    recon = df["reconstruction_error"].fillna(0)
    _style_chart_ax(ax)
    threshold_path = "ml/model/threshold.pkl"
    if os.path.exists(threshold_path):
        with open(threshold_path, "rb") as f:
            threshold = pickle.load(f)
    else:
        threshold = df["reconstruction_error"].quantile(0.95)
    ax.fill_between(row_index, recon, 0, color="#ce93d8", alpha=0.25)
    ax.plot(row_index, recon, color="#ab47bc", linewidth=1)
    ax.axhline(threshold, color="#ff5252", linestyle="--", linewidth=1.5, label="Threshold")
    ax.set_title("LSTM Reconstruction Error", color="#e0e0e0", fontsize=11)
    ax.set_ylabel("MAE")
    ax.set_xlabel("Reading Index")
    ax.legend(fontsize=8, facecolor="#1a1d2e", edgecolor="#2a2d3e", labelcolor="white")
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_compliance_chart(df):
    fig, ax = plt.subplots(figsize=(11, 3.8), facecolor="#0f1117")
    row_index = np.arange(len(df))
    _style_chart_ax(ax)
    ax.axhspan(85, 100, color="#69f0ae", alpha=0.08)
    ax.axhspan(0, 85, color="#ef5350", alpha=0.08)
    ax.plot(row_index, df["compliance_score"], color="#69f0ae", linewidth=1.2)
    ax.fill_between(row_index, df["compliance_score"], 0, color="#69f0ae", alpha=0.15)
    ax.axhline(85, color="#ffb74d", linestyle="--", linewidth=1, label="Pass Threshold")
    ax.set_title("Compliance Score Trend", color="#e0e0e0", fontsize=11)
    ax.set_ylabel("Score (%)")
    ax.set_xlabel("Reading Index")
    ax.legend(fontsize=8, facecolor="#1a1d2e", edgecolor="#2a2d3e", labelcolor="white")
    plt.tight_layout()
    return _fig_to_base64(fig)


TRANSLATIONS = {
    "en": {
        "dashboard": "Dashboard", "overview": "Overview", "temperature": "Temperature",
        "anomalies": "Anomalies", "gps": "GPS Tracking", "reports": "Reports",
        "settings": "Settings", "logout": "Logout", "avg_temp": "Avg Temperature",
        "peak_temp": "Peak Temperature", "compliance": "Compliance Score", "status": "Status",
        "alerts": "Total Alerts", "anomalies_detected": "Anomalies Detected",
        "spoilage_risk": "Peak Spoilage Risk", "door_events": "Door Events",
        "shipment": "Shipment", "driver": "Driver", "vehicle": "Vehicle",
        "start": "Start", "end": "End", "readings": "Total Readings",
        "welcome": "Welcome back", "good_morning": "Good Morning",
        "system_normal": "All systems normal", "excursion_warning": "Temperature excursion detected",
        "pass": "PASS", "fail": "FAIL", "dark_mode": "Dark Mode", "light_mode": "Light Mode",
        "language": "Language", "notifications": "Notifications", "email_alerts": "Email Alerts",
        "telegram_alerts": "Telegram Alerts", "alert_threshold": "Alert Threshold (°C)",
        "save_settings": "Save Settings", "general": "General Settings", "appearance": "Appearance",
        "account": "Account", "username": "Username", "role": "Role", "version": "System Version",
        "temp_timeline": "Temperature Timeline", "spoilage_prob": "Spoilage Probability",
        "humidity": "Humidity Monitoring", "gps_route": "GPS Route Map",
        "lstm_error": "LSTM Reconstruction Error", "compliance_trend": "Compliance Score Trend",
        "no_alerts": "No active alerts. All readings within safe range.",
        "alert_message": "temperature excursion alerts detected",
        "excursion_min": "Excursion Duration", "duration": "Delivery Duration",
        "battery": "Battery Level", "signal": "Signal Strength",
    },
    "ta": {
        "dashboard": "டாஷ்போர்டு", "overview": "கண்ணோட்டம்", "temperature": "வெப்பநிலை",
        "anomalies": "முரண்பாடுகள்", "gps": "GPS கண்காணிப்பு", "reports": "அறிக்கைகள்",
        "settings": "அமைப்புகள்", "logout": "வெளியேறு", "avg_temp": "சராசரி வெப்பநிலை",
        "peak_temp": "உச்ச வெப்பநிலை", "compliance": "இணக்க மதிப்பெண்", "status": "நிலை",
        "alerts": "மொத்த எச்சரிக்கைகள்", "anomalies_detected": "கண்டறியப்பட்ட முரண்பாடுகள்",
        "spoilage_risk": "கெடுதல் அபாயம்", "door_events": "கதவு நிகழ்வுகள்",
        "shipment": "ஏற்றுமதி", "driver": "ஓட்டுனர்", "vehicle": "வாகனம்",
        "start": "தொடக்கம்", "end": "முடிவு", "readings": "மொத்த அளவீடுகள்",
        "welcome": "மீண்டும் வரவேற்கிறோம்", "good_morning": "காலை வணக்கம்",
        "system_normal": "அனைத்து அமைப்புகளும் சரியாக உள்ளன",
        "excursion_warning": "வெப்பநிலை விலகல் கண்டறியப்பட்டது",
        "pass": "தேர்ச்சி", "fail": "தோல்வி", "dark_mode": "இருண்ட பயன்முறை",
        "light_mode": "வெளிர் பயன்முறை", "language": "மொழி", "notifications": "அறிவிப்புகள்",
        "email_alerts": "மின்னஞ்சல் எச்சரிக்கைகள்", "telegram_alerts": "டெலிகிராம் எச்சரிக்கைகள்",
        "alert_threshold": "எச்சரிக்கை வரம்பு (°C)", "save_settings": "அமைப்புகளை சேமி",
        "general": "பொது அமைப்புகள்", "appearance": "தோற்றம்", "account": "கணக்கு",
        "username": "பயனர்பெயர்", "role": "பங்கு", "version": "கணினி பதிப்பு",
        "temp_timeline": "வெப்பநிலை காலவரிசை", "spoilage_prob": "கெடுதல் நிகழ்தகவு",
        "humidity": "ஈரப்பதம் கண்காணிப்பு", "gps_route": "GPS பாதை வரைபடம்",
        "lstm_error": "LSTM மறுகட்டமைப்பு பிழை", "compliance_trend": "இணக்க மதிப்பெண் போக்கு",
        "no_alerts": "செயலில் உள்ள எச்சரிக்கைகள் இல்லை.",
        "alert_message": "வெப்பநிலை விலகல் எச்சரிக்கைகள்",
        "excursion_min": "விலகல் கால அளவு", "duration": "டெலிவரி கால அளவு",
        "battery": "பேட்டரி நிலை", "signal": "சிக்னல் வலிமை",
    },
    "te": {
        "dashboard": "డాష్‌బోర్డ్", "overview": "అవలోకనం", "temperature": "ఉష్ణోగ్రత",
        "anomalies": "క్రమరాహిత్యాలు", "gps": "GPS ట్రాకింగ్", "reports": "నివేదికలు",
        "settings": "సెట్టింగులు", "logout": "లాగ్అవుట్", "avg_temp": "సగటు ఉష్ణోగ్రత",
        "peak_temp": "గరిష్ట ఉష్ణోగ్రత", "compliance": "సమ్మతి స్కోర్", "status": "స్థితి",
        "alerts": "మొత్తం హెచ్చరికలు", "anomalies_detected": "గుర్తించిన క్రమరాహిత్యాలు",
        "spoilage_risk": "నాశనం ప్రమాదం", "door_events": "తలుపు సంఘటనలు",
        "shipment": "రవాణా", "driver": "డ్రైవర్", "vehicle": "వాహనం",
        "start": "ప్రారంభం", "end": "ముగింపు", "readings": "మొత్తం రీడింగులు",
        "welcome": "తిరిగి స్వాగతం", "good_morning": "శుభోదయం",
        "system_normal": "అన్ని వ్యవస్థలు సాధారణంగా ఉన్నాయి",
        "excursion_warning": "ఉష్ణోగ్రత విచలనం గుర్తించబడింది",
        "pass": "పాస్", "fail": "ఫెయిల్", "dark_mode": "డార్క్ మోడ్",
        "light_mode": "లైట్ మోడ్", "language": "భాష", "notifications": "నోటిఫికేషన్లు",
        "email_alerts": "ఇమెయిల్ హెచ్చరికలు", "telegram_alerts": "టెలిగ్రామ్ హెచ్చరికలు",
        "alert_threshold": "హెచ్చరిక పరిమితి (°C)", "save_settings": "సెట్టింగులు సేవ్ చేయి",
        "general": "సాధారణ సెట్టింగులు", "appearance": "రూపం", "account": "ఖాతా",
        "username": "వినియోగదారు పేరు", "role": "పాత్ర", "version": "సిస్టమ్ వెర్షన్",
        "temp_timeline": "ఉష్ణోగ్రత టైమ్‌లైన్", "spoilage_prob": "నాశనం సంభావ్యత",
        "humidity": "తేమ పర్యవేక్షణ", "gps_route": "GPS మార్గం",
        "lstm_error": "LSTM పునర్నిర్మాణ లోపం", "compliance_trend": "సమ్మతి స్కోర్ ధోరణి",
        "no_alerts": "క్రియాశీల హెచ్చరికలు లేవు.",
        "alert_message": "ఉష్ణోగ్రత విచలన హెచ్చరికలు",
        "excursion_min": "విచలన వ్యవధి", "duration": "డెలివరీ వ్యవధి",
        "battery": "బ్యాటరీ స్థాయి", "signal": "సిగ్నల్ బలం",
    },
}


def get_t(lang, key):
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, key)


LOGIN_CSS = """* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #0f1117; font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; }
.login-page {
  min-height: 100vh;
  background: #0f1117;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px;
}
.login-container {
  display: flex;
  width: 100%;
  max-width: 900px;
  min-height: 520px;
  border-radius: 20px;
  overflow: hidden;
  box-shadow: 0 24px 80px rgba(0,0,0,0.5);
}
.login-left {
  flex: 1;
  background: linear-gradient(135deg, #0d47a1 0%, #1565c0 50%, #0288d1 100%);
  padding: 48px 40px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  position: relative;
  overflow: hidden;
}
.login-left::before {
  content: '';
  position: absolute;
  top: -40px;
  right: -40px;
  width: 200px;
  height: 200px;
  background: rgba(255,255,255,0.05);
  border-radius: 50%;
}
.login-left::after {
  content: '';
  position: absolute;
  bottom: -60px;
  left: -30px;
  width: 250px;
  height: 250px;
  background: rgba(255,255,255,0.04);
  border-radius: 50%;
}
.login-brand .brand-icon { font-size: 40px; display: block; margin-bottom: 12px; }
.login-brand .brand-name { font-size: 22px; font-weight: 800; color: white; letter-spacing: 0.3px; }
.login-brand .brand-tagline { font-size: 13px; color: rgba(255,255,255,0.6); margin-top: 6px; line-height: 1.5; }
.login-features { list-style: none; }
.login-features li {
  display: flex;
  align-items: center;
  gap: 10px;
  color: rgba(255,255,255,0.75);
  font-size: 13px;
  margin-bottom: 12px;
}
.login-features li::before { content: '\2713'; color: #69f0ae; font-weight: 700; }
.login-compliance { display: flex; gap: 8px; flex-wrap: wrap; }
.compliance-badge {
  background: rgba(255,255,255,0.1);
  color: rgba(255,255,255,0.8);
  padding: 4px 10px;
  border-radius: 12px;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.5px;
  border: 1px solid rgba(255,255,255,0.15);
}
.login-right {
  width: 380px;
  background: #1a1d2e;
  padding: 48px 40px;
  display: flex;
  flex-direction: column;
  justify-content: center;
}
.login-right h2 { font-size: 22px; font-weight: 700; color: white; margin-bottom: 6px; }
.login-right .login-sub { font-size: 13px; color: #8b8fa8; margin-bottom: 32px; }
.form-group { margin-bottom: 18px; }
.form-label {
  display: block;
  font-size: 12px;
  font-weight: 600;
  color: #8b8fa8;
  text-transform: uppercase;
  letter-spacing: 0.6px;
  margin-bottom: 8px;
}
.form-input {
  width: 100%;
  background: #252840;
  border: 1px solid #2a2d3e;
  color: white;
  padding: 12px 16px;
  border-radius: 10px;
  font-size: 14px;
  outline: none;
  transition: border-color 0.15s;
}
.form-input:focus { border-color: #4fc3f7; }
.login-btn {
  width: 100%;
  background: linear-gradient(135deg, #1976d2, #0d47a1);
  color: white;
  border: none;
  padding: 13px;
  border-radius: 10px;
  font-size: 15px;
  font-weight: 600;
  cursor: pointer;
  margin-top: 8px;
  transition: opacity 0.15s;
  letter-spacing: 0.3px;
}
.login-btn:hover { opacity: 0.88; }
.login-error {
  background: rgba(255,82,82,0.1);
  border: 1px solid rgba(255,82,82,0.25);
  color: #ff5252;
  padding: 10px 14px;
  border-radius: 8px;
  font-size: 12px;
  margin-bottom: 16px;
}
.demo-hint { margin-top: 20px; text-align: center; font-size: 11px; color: #5c6080; }
.demo-hint span { color: #4fc3f7; font-weight: 600; }"""

DARK_CSS_ROOT = """--bg-primary: #0f1117;
--bg-secondary: #1a1d2e;
--bg-card: #1e2130;
--bg-sidebar: #13151f;
--bg-header: #0d0f1a;
--border: #2a2d3e;
--text-primary: #e8eaf6;
--text-secondary: #8b8fa8;
--text-muted: #5c6080;
--accent-blue: #4fc3f7;
--accent-green: #69f0ae;
--accent-red: #ff5252;
--accent-orange: #ffb74d;
--accent-purple: #ce93d8;
--shadow: rgba(0,0,0,0.4);
--input-bg: #252840;"""
LIGHT_CSS_ROOT = """--bg-primary: #f0f2f8;
--bg-secondary: #ffffff;
--bg-card: #ffffff;
--bg-sidebar: #1a237e;
--bg-header: #1a237e;
--border: #e0e4f0;
--text-primary: #1a1d2e;
--text-secondary: #5c6080;
--text-muted: #9e9eb8;
--accent-blue: #1976d2;
--accent-green: #2e7d32;
--accent-red: #c62828;
--accent-orange: #e65100;
--accent-purple: #6a1b9a;
--shadow: rgba(0,0,0,0.1);
--input-bg: #f5f7ff;"""
DASHBOARD_CSS = """* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  background: var(--bg-primary);
  color: var(--text-primary);
  font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
  font-size: 14px;
  line-height: 1.5;
  overflow-x: hidden;
}
.app-layout { display: flex; min-height: 100vh; }
.sidebar {
  width: 240px;
  min-height: 100vh;
  background: var(--bg-sidebar);
  display: flex;
  flex-direction: column;
  position: fixed;
  left: 0;
  top: 0;
  bottom: 0;
  z-index: 100;
  border-right: 1px solid var(--border);
  box-shadow: 4px 0 20px var(--shadow);
}
.sidebar-logo { padding: 24px 20px 20px; border-bottom: 1px solid rgba(255,255,255,0.08); }
.sidebar-logo .logo-icon { font-size: 28px; display: block; margin-bottom: 6px; }
.sidebar-logo .logo-text { font-size: 15px; font-weight: 700; color: white; letter-spacing: 0.3px; }
.sidebar-logo .logo-sub {
  font-size: 10px;
  color: rgba(255,255,255,0.45);
  margin-top: 2px;
  letter-spacing: 0.5px;
  text-transform: uppercase;
}
.sidebar-nav { flex: 1; padding: 16px 12px; overflow-y: auto; }
.nav-section-label {
  font-size: 10px;
  font-weight: 600;
  color: rgba(255,255,255,0.3);
  text-transform: uppercase;
  letter-spacing: 1.2px;
  padding: 8px 8px 6px;
  margin-top: 8px;
}
.nav-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 12px;
  border-radius: 8px;
  cursor: pointer;
  color: rgba(255,255,255,0.6);
  font-size: 13px;
  font-weight: 500;
  margin-bottom: 2px;
  transition: all 0.15s ease;
  text-decoration: none;
  border: none;
  background: transparent;
  width: 100%;
  text-align: left;
}
.nav-item:hover { background: rgba(255,255,255,0.08); color: rgba(255,255,255,0.9); }
.nav-item.active { background: rgba(79,195,247,0.15); color: #4fc3f7; font-weight: 600; }
.nav-item .nav-icon { font-size: 16px; width: 20px; text-align: center; }
.nav-item .nav-badge {
  margin-left: auto;
  background: var(--accent-red);
  color: white;
  font-size: 10px;
  padding: 2px 6px;
  border-radius: 10px;
  font-weight: 600;
}
.sidebar-user {
  padding: 16px;
  border-top: 1px solid rgba(255,255,255,0.08);
  display: flex;
  align-items: center;
  gap: 10px;
}
.user-avatar {
  width: 36px;
  height: 36px;
  background: linear-gradient(135deg, #4fc3f7, #1976d2);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: 700;
  color: white;
  flex-shrink: 0;
}
.user-info .user-name { font-size: 12px; font-weight: 600; color: white; }
.user-info .user-role { font-size: 10px; color: rgba(255,255,255,0.4); }
.main-content { margin-left: 240px; flex: 1; display: flex; flex-direction: column; min-height: 100vh; }
.top-header {
  background: var(--bg-header);
  border-bottom: 1px solid var(--border);
  padding: 0 32px;
  height: 64px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  position: sticky;
  top: 0;
  z-index: 50;
  box-shadow: 0 2px 12px var(--shadow);
}
.header-left .page-title { font-size: 18px; font-weight: 700; color: var(--text-primary); }
.header-left .page-subtitle { font-size: 11px; color: var(--text-secondary); margin-top: 1px; }
.header-right { display: flex; align-items: center; gap: 16px; }
.status-pill {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 14px;
  border-radius: 20px;
  font-size: 12px;
  font-weight: 600;
}
.status-pill.online {
  background: rgba(105,240,174,0.12);
  color: #69f0ae;
  border: 1px solid rgba(105,240,174,0.25);
}
.status-pill.alert {
  background: rgba(255,82,82,0.12);
  color: #ff5252;
  border: 1px solid rgba(255,82,82,0.25);
}
.status-pill .dot { width: 7px; height: 7px; border-radius: 50%; background: #69f0ae; animation: pulse 2s infinite; }
.status-pill.alert .dot { background: #ff5252; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
.header-btn {
  background: var(--bg-card);
  border: 1px solid var(--border);
  color: var(--text-primary);
  padding: 8px 16px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 12px;
  font-weight: 500;
  transition: all 0.15s;
  text-decoration: none;
  display: inline-block;
}
.header-btn:hover { background: var(--bg-secondary); }
.page-content { flex: 1; padding: 28px 32px; background: var(--bg-primary); }
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
  margin-bottom: 24px;
}
.kpi-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 20px;
  position: relative;
  overflow: hidden;
  transition: transform 0.15s, box-shadow 0.15s;
}
.kpi-card:hover { transform: translateY(-2px); box-shadow: 0 8px 24px var(--shadow); }
.kpi-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; }
.kpi-card.good::before { background: linear-gradient(90deg, #69f0ae, #00c853); }
.kpi-card.warn::before { background: linear-gradient(90deg, #ffb74d, #ff6f00); }
.kpi-card.bad::before { background: linear-gradient(90deg, #ff5252, #b71c1c); }
.kpi-card.info::before { background: linear-gradient(90deg, #4fc3f7, #0288d1); }
.kpi-card.purple::before { background: linear-gradient(90deg, #ce93d8, #6a1b9a); }
.kpi-icon { font-size: 22px; margin-bottom: 10px; display: block; }
.kpi-label {
  font-size: 11px;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.8px;
  margin-bottom: 6px;
  font-weight: 500;
}
.kpi-value { font-size: 26px; font-weight: 700; line-height: 1; }
.kpi-value.good { color: #69f0ae; }
.kpi-value.warn { color: #ffb74d; }
.kpi-value.bad { color: #ff5252; }
.kpi-value.info { color: #4fc3f7; }
.kpi-value.purple { color: #ce93d8; }
.kpi-sub { font-size: 11px; color: var(--text-muted); margin-top: 6px; }
.charts-grid-2col { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
.chart-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 20px;
  box-shadow: 0 2px 8px var(--shadow);
}
.chart-card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; }
.chart-card-title { font-size: 13px; font-weight: 600; color: var(--text-primary); }
.chart-card-badge { font-size: 10px; padding: 3px 8px; border-radius: 10px; font-weight: 600; }
.badge-live { background: rgba(105,240,174,0.15); color: #69f0ae; }
.badge-ml { background: rgba(206,147,216,0.15); color: #ce93d8; }
.badge-gps { background: rgba(79,195,247,0.15); color: #4fc3f7; }
.badge-warn { background: rgba(255,183,77,0.15); color: #ffb74d; }
.chart-card img { width: 100%; border-radius: 8px; display: block; }
.info-strip {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px 24px;
  display: flex;
  gap: 40px;
  margin-bottom: 24px;
  flex-wrap: wrap;
  align-items: center;
}
.info-strip-item { display: flex; flex-direction: column; gap: 3px; }
.info-strip-label {
  font-size: 10px;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.8px;
}
.info-strip-value { font-size: 13px; font-weight: 600; color: var(--text-primary); }
.alert-banner {
  border-radius: 10px;
  padding: 14px 20px;
  margin-bottom: 20px;
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 13px;
  font-weight: 500;
}
.alert-banner.safe {
  background: rgba(105,240,174,0.08);
  border: 1px solid rgba(105,240,174,0.2);
  color: #69f0ae;
}
.alert-banner.danger {
  background: rgba(255,82,82,0.08);
  border: 1px solid rgba(255,82,82,0.2);
  color: #ff5252;
}
.table-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
  margin-bottom: 20px;
}
.table-card-header {
  padding: 16px 20px;
  border-bottom: 1px solid var(--border);
  font-size: 13px;
  font-weight: 600;
  color: var(--text-primary);
  display: flex;
  justify-content: space-between;
  align-items: center;
}
table { width: 100%; border-collapse: collapse; }
thead th {
  padding: 10px 16px;
  text-align: left;
  font-size: 11px;
  font-weight: 600;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.6px;
  background: var(--bg-secondary);
  border-bottom: 1px solid var(--border);
}
tbody tr { border-bottom: 1px solid var(--border); transition: background 0.1s; }
tbody tr:hover { background: rgba(255,255,255,0.03); }
tbody tr:last-child { border-bottom: none; }
tbody td { padding: 10px 16px; font-size: 12px; color: var(--text-primary); }
.td-badge {
  display: inline-flex;
  align-items: center;
  padding: 3px 8px;
  border-radius: 6px;
  font-size: 11px;
  font-weight: 600;
}
.td-badge.pass { background: rgba(105,240,174,0.12); color: #69f0ae; }
.td-badge.fail { background: rgba(255,82,82,0.12); color: #ff5252; }
.td-badge.warn { background: rgba(255,183,77,0.12); color: #ffb74d; }
.td-badge.normal { background: rgba(79,195,247,0.12); color: #4fc3f7; }
.settings-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.settings-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 24px;
}
.settings-card-title {
  font-size: 14px;
  font-weight: 700;
  color: var(--text-primary);
  margin-bottom: 20px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 8px;
}
.settings-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 0;
  border-bottom: 1px solid var(--border);
}
.settings-row:last-child { border-bottom: none; }
.settings-label { font-size: 13px; color: var(--text-primary); font-weight: 500; }
.settings-sublabel { font-size: 11px; color: var(--text-muted); margin-top: 2px; }
.toggle-switch { position: relative; width: 44px; height: 24px; }
.toggle-switch input { opacity: 0; width: 0; height: 0; }
.toggle-slider {
  position: absolute;
  cursor: pointer;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: var(--border);
  border-radius: 24px;
  transition: 0.3s;
}
.toggle-slider:before {
  content: '';
  position: absolute;
  width: 18px;
  height: 18px;
  left: 3px;
  bottom: 3px;
  background: white;
  border-radius: 50%;
  transition: 0.3s;
}
input:checked + .toggle-slider { background: #4fc3f7; }
input:checked + .toggle-slider:before { transform: translateX(20px); }
.select-input {
  background: var(--input-bg);
  border: 1px solid var(--border);
  color: var(--text-primary);
  padding: 7px 12px;
  border-radius: 8px;
  font-size: 12px;
  cursor: pointer;
  outline: none;
}
.number-input {
  background: var(--input-bg);
  border: 1px solid var(--border);
  color: var(--text-primary);
  padding: 7px 12px;
  border-radius: 8px;
  font-size: 12px;
  width: 80px;
  outline: none;
}
.save-btn {
  background: linear-gradient(135deg, #1976d2, #0d47a1);
  color: white;
  border: none;
  padding: 10px 24px;
  border-radius: 8px;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  margin-top: 20px;
  transition: opacity 0.15s;
}
.save-btn:hover { opacity: 0.88; }
.reports-summary {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 24px;
  margin-bottom: 20px;
}
.reports-summary-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px 32px; }
.reports-summary-item { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid var(--border); }
.reports-summary-item:last-child { border-bottom: none; }
.reports-note {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px 20px;
  margin-bottom: 20px;
  font-size: 13px;
  color: var(--text-secondary);
}
.timeline-row { display: flex; justify-content: space-between; padding: 12px 16px; border-bottom: 1px solid var(--border); font-size: 12px; }
.timeline-row:last-child { border-bottom: none; }
.full-width-chart { grid-column: 1 / -1; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }"""


def get_css(theme, lang):
    root = DARK_CSS_ROOT if theme != "light" else LIGHT_CSS_ROOT
    return ":root {\n" + root + "\n}\n" + DASHBOARD_CSS + "\n" + LOGIN_CSS


def _esc(s):
    if s is None:
        return ""
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _q(tab, lang, theme):
    return f"/?tab={tab}&lang={lang}&theme={theme}"


def _kpi_card(icon, label, value, card_class, value_class, sub=""):
    sub_html = f'<div class="kpi-sub">{_esc(sub)}</div>' if sub else ""
    return f"""<div class="kpi-card {card_class}">
<span class="kpi-icon">{icon}</span>
<div class="kpi-label">{_esc(label)}</div>
<div class="kpi-value {value_class}">{_esc(value)}</div>
{sub_html}
</div>"""


def _chart_card(title, img_src, badge_class, badge_text):
    return f"""<div class="chart-card">
<div class="chart-card-header">
<span class="chart-card-title">{_esc(title)}</span>
<span class="chart-card-badge {badge_class}">{_esc(badge_text)}</span>
</div>
<img src="{img_src}" alt="{_esc(title)}">
</div>"""

TAB_TITLES = {
    "overview": "overview",
    "temperature": "temperature",
    "anomalies": "anomalies",
    "gps": "gps",
    "reports": "reports",
    "settings": "settings",
}

TAB_ICONS = {
    "overview": "&#128202;",
    "temperature": "&#127777;",
    "anomalies": "&#128300;",
    "gps": "&#128205;",
    "reports": "&#128196;",
    "settings": "&#9881;",
}


def build_sidebar(tab, kpis, user, lang, theme):
    nav_main = [
        ("overview", TAB_ICONS["overview"]),
        ("temperature", TAB_ICONS["temperature"]),
        ("anomalies", TAB_ICONS["anomalies"]),
        ("gps", TAB_ICONS["gps"]),
        ("reports", TAB_ICONS["reports"]),
    ]
    items = []
    for tkey, icon in nav_main:
        active = " active" if tab == tkey else ""
        badge = ""
        if tkey == "anomalies" and kpis.get("total_alerts", 0) > 0:
            badge = f'<span class="nav-badge">{kpis["total_alerts"]}</span>'
        items.append(
            f'<a class="nav-item{active}" href="{_q(tkey, lang, theme)}">'
            f'<span class="nav-icon">{icon}</span>'
            f'<span>{get_t(lang, tkey)}</span>{badge}</a>'
        )
    settings_active = " active" if tab == "settings" else ""
    settings_item = (
        f'<a class="nav-item{settings_active}" href="{_q("settings", lang, theme)}">'
        f'<span class="nav-icon">{TAB_ICONS["settings"]}</span>'
        f'<span>{get_t(lang, "settings")}</span></a>'
    )
    logout_item = (
        '<a class="nav-item" href="/logout">'
        '<span class="nav-icon">&#128682;</span>'
        f'<span>{get_t(lang, "logout")}</span></a>'
    )
    return f"""<aside class="sidebar">
<div class="sidebar-logo">
<span class="logo-icon">&#127777;</span>
<div class="logo-text">ColdChain Monitor</div>
<div class="logo-sub">Pharmaceutical Logistics</div>
</div>
<nav class="sidebar-nav">
<div class="nav-section-label">MAIN MENU</div>
{"".join(items)}
<div class="nav-section-label">SYSTEM</div>
{settings_item}
{logout_item}
</nav>
<div class="sidebar-user">
<div class="user-avatar">{_esc(user.get("avatar", "U"))}</div>
<div class="user-info">
<div class="user-name">{_esc(user.get("name", ""))}</div>
<div class="user-role">{_esc(user.get("role", ""))}</div>
</div>
</div>
</aside>"""


def build_header(tab, kpis, user, lang, theme):
    title = get_t(lang, TAB_TITLES.get(tab, "overview"))
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if kpis.get("total_alerts", 0) > 0:
        pill = (
            f'<div class="status-pill alert">'
            f'<span class="dot"></span>'
            f'{kpis["total_alerts"]} ALERTS</div>'
        )
    else:
        pill = '<div class="status-pill online"><span class="dot"></span>LIVE</div>'
    if theme == "dark":
        toggle_href = _q(tab, lang, "light")
        toggle_text = "Light"
        toggle_icon = "&#9728;"
    else:
        toggle_href = _q(tab, lang, "dark")
        toggle_text = "Dark"
        toggle_icon = "&#127769;"
    return f"""<header class="top-header">
<div class="header-left">
<div class="page-title">{_esc(title)}</div>
<div class="page-subtitle">{now_str}</div>
</div>
<div class="header-right">
{pill}
<a class="header-btn" href="{toggle_href}">{toggle_icon} {toggle_text}</a>
</div>
</header>"""


def build_kpi_row(kpis, lang):
    avg_ok = TEMP_MIN <= kpis["avg_temp"] <= TEMP_MAX
    peak_ok = kpis["max_temp"] <= TEMP_MAX
    comp_ok = kpis["final_compliance"] >= 85
    status_val = get_t(lang, "pass") if kpis["compliance_status"] == "PASS" else get_t(lang, "fail")
    alerts_bad = kpis["total_alerts"] > 0
    spoil = kpis["peak_spoilage"]
    if spoil > 80:
        spoil_v, spoil_c, spoil_card = f"{spoil}%", "bad", "bad"
    elif spoil > 50:
        spoil_v, spoil_c, spoil_card = f"{spoil}%", "warn", "warn"
    else:
        spoil_v, spoil_c, spoil_card = f"{spoil}%", "good", "good"
    bat = kpis["battery_level"]
    if bat > 50:
        bat_v, bat_c, bat_card = f"{bat}%", "good", "good"
    elif bat > 20:
        bat_v, bat_c, bat_card = f"{bat}%", "warn", "warn"
    else:
        bat_v, bat_c, bat_card = f"{bat}%", "bad", "bad"
    cards = [
        _kpi_card("&#127777;", get_t(lang, "avg_temp"), f"{kpis['avg_temp']}C",
                  "good" if avg_ok else "bad", "good" if avg_ok else "bad",
                  f"Safe: {TEMP_MIN}-{TEMP_MAX}C"),
        _kpi_card("&#128314;", get_t(lang, "peak_temp"), f"{kpis['max_temp']}C",
                  "good" if peak_ok else "bad", "good" if peak_ok else "bad",
                  f"Min: {kpis['min_temp']}C"),
        _kpi_card("&#9989;", get_t(lang, "compliance"), f"{kpis['final_compliance']}%",
                  "good" if comp_ok else "bad", "good" if comp_ok else "bad",
                  kpis["compliance_status"]),
        _kpi_card("&#128203;", get_t(lang, "status"), status_val,
                  "good" if comp_ok else "bad", "good" if comp_ok else "bad",
                  f"Score: {kpis['final_compliance']}%"),
        _kpi_card("&#128680;", get_t(lang, "alerts"), str(kpis["total_alerts"]),
                  "bad" if alerts_bad else "good", "bad" if alerts_bad else "good",
                  f"{kpis['anomaly_count']} anomalies"),
        _kpi_card("&#9763;", get_t(lang, "spoilage_risk"), spoil_v,
                  spoil_card, spoil_c, ""),
        _kpi_card("&#128682;", get_t(lang, "door_events"), str(kpis["total_door_events"]),
                  "info", "info", "Duration tracked"),
        _kpi_card("&#128267;", get_t(lang, "battery"), bat_v, bat_card, bat_c, ""),
    ]
    return '<div class="kpi-grid">' + "".join(cards) + "</div>"


def build_info_strip(kpis, lang):
    items = [
        ("shipment", kpis["shipment_id"]),
        ("driver", kpis["driver"]),
        ("vehicle", kpis["vehicle"]),
        ("start", kpis["delivery_start"]),
        ("end", kpis["delivery_end"]),
        ("readings", str(kpis["total_rows"])),
    ]
    html = ""
    for key, val in items:
        html += f"""<div class="info-strip-item">
<div class="info-strip-label">{get_t(lang, key)}</div>
<div class="info-strip-value">{_esc(val)}</div>
</div>"""
    return f'<div class="info-strip">{html}</div>'


def build_alert_banner(kpis, lang):
    if kpis.get("total_alerts", 0) == 0:
        return f'<div class="alert-banner safe">&#9989; {get_t(lang, "no_alerts")}</div>'
    return (
        f'<div class="alert-banner danger">&#9888; {kpis["total_alerts"]} '
        f'{get_t(lang, "alert_message")} (peak {kpis["max_temp"]}C)</div>'
    )


def build_overview_tab(kpis, charts, lang):
    grid = f"""<div class="charts-grid-2col">
{_chart_card(get_t(lang, "temp_timeline"), charts["temp"], "badge-live", "LIVE")}
{_chart_card(get_t(lang, "spoilage_prob"), charts["spoilage"], "badge-ml", "ML")}
{_chart_card(get_t(lang, "humidity"), charts["humidity"], "badge-warn", "SENSOR")}
{_chart_card(get_t(lang, "gps_route"), charts["gps"], "badge-gps", "GPS")}
</div>"""
    return build_alert_banner(kpis, lang) + build_kpi_row(kpis, lang) + grid


def _bool_badge(val, true_label="YES", false_label="NO", true_class="fail", false_class="pass"):
    if val:
        return f'<span class="td-badge {true_class}">{true_label}</span>'
    return f'<span class="td-badge {false_class}">{false_label}</span>'


def build_temperature_tab(df, kpis, charts, lang):
    avg_ok = TEMP_MIN <= kpis["avg_temp"] <= TEMP_MAX
    peak_ok = kpis["max_temp"] <= TEMP_MAX
    exc = kpis["excursion_minutes"]
    exc_class = "bad" if exc > 30 else ("warn" if exc > 10 else "good")
    kpis3 = f"""<div class="kpi-grid">
{_kpi_card("&#127777;", get_t(lang, "avg_temp"), f"{kpis['avg_temp']}C", "good" if avg_ok else "bad", "good" if avg_ok else "bad", "")}
{_kpi_card("&#128314;", get_t(lang, "peak_temp"), f"{kpis['max_temp']}C", "good" if peak_ok else "bad", "good" if peak_ok else "bad", "")}
{_kpi_card("&#9201;", get_t(lang, "excursion_min"), f"{exc} min", exc_class, exc_class, "")}
</div>"""
    top = df.nlargest(15, "temperature")
    rows = ""
    for _, row in top.iterrows():
        rows += f"""<tr>
<td>{_esc(row['timestamp'])}</td>
<td>{row['temperature']:.2f}C</td>
<td>{row['humidity']:.1f}%</td>
<td>{_bool_badge(bool(row.get('is_anomaly', False)))}</td>
<td>{_bool_badge(bool(row.get('alert_triggered', False)))}</td>
</tr>"""
    table = f"""<div class="table-card">
<div class="table-card-header">Top 15 Temperature Readings</div>
<table><thead><tr>
<th>Timestamp</th><th>Temperature</th><th>Humidity</th><th>Anomaly</th><th>Alert</th>
</tr></thead><tbody>{rows}</tbody></table></div>"""
    c1 = _chart_card(get_t(lang, "temp_timeline"), charts["temp"], "badge-live", "LIVE").replace('class="chart-card"', 'class="chart-card full-width-chart"', 1)
    c2 = _chart_card(get_t(lang, "humidity"), charts["humidity"], "badge-warn", "SENSOR").replace('class="chart-card"', 'class="chart-card full-width-chart"', 1)
    return build_alert_banner(kpis, lang) + kpis3 + c1 + c2 + table


def build_anomalies_tab(df, kpis, charts, lang):
    spoil = kpis["peak_spoilage"]
    if spoil > 80:
        spoil_v, spoil_c, spoil_card = f"{spoil}%", "bad", "bad"
    elif spoil > 50:
        spoil_v, spoil_c, spoil_card = f"{spoil}%", "warn", "warn"
    else:
        spoil_v, spoil_c, spoil_card = f"{spoil}%", "good", "good"
    alerts_bad = kpis["total_alerts"] > 0
    kpis3 = f"""<div class="kpi-grid">
{_kpi_card("&#128300;", get_t(lang, "anomalies_detected"), str(kpis["anomaly_count"]), "bad", "bad", "")}
{_kpi_card("&#9763;", get_t(lang, "spoilage_risk"), spoil_v, spoil_card, spoil_c, "")}
{_kpi_card("&#128680;", get_t(lang, "alerts"), str(kpis["total_alerts"]), "bad" if alerts_bad else "good", "bad" if alerts_bad else "good", "")}
</div>"""
    top = df.nlargest(15, "reconstruction_error")
    rows = ""
    for _, row in top.iterrows():
        spoil_pct = float(row.get("ml_spoilage_probability", 0)) * 100
        rows += f"""<tr>
<td>{_esc(row['timestamp'])}</td>
<td>{row['temperature']:.2f}C</td>
<td>{float(row.get('reconstruction_error', 0)):.4f}</td>
<td>{spoil_pct:.1f}%</td>
<td>{_bool_badge(bool(row.get('alert_triggered', False)), "ALERT", "OK")}</td>
</tr>"""
    table = f"""<div class="table-card">
<div class="table-card-header">Top 15 Anomaly Readings</div>
<table><thead><tr>
<th>Timestamp</th><th>Temperature</th><th>Reconstruction Error</th><th>Spoilage Risk</th><th>Alert Status</th>
</tr></thead><tbody>{rows}</tbody></table></div>"""
    c1 = _chart_card(get_t(lang, "spoilage_prob"), charts["spoilage"], "badge-ml", "ML").replace('class="chart-card"', 'class="chart-card full-width-chart"', 1)
    c2 = _chart_card(get_t(lang, "lstm_error"), charts["reconstruction"], "badge-ml", "LSTM").replace('class="chart-card"', 'class="chart-card full-width-chart"', 1)
    return build_alert_banner(kpis, lang) + kpis3 + c1 + c2 + table


def build_gps_tab(df, kpis, charts, lang):
    kpis3 = f"""<div class="kpi-grid">
{_kpi_card("&#128663;", "Avg Speed", f"{kpis['avg_speed']} km/h", "info", "info", "")}
{_kpi_card("&#128682;", get_t(lang, "door_events"), str(kpis["total_door_events"]), "info", "info", "")}
{_kpi_card("&#9201;", get_t(lang, "duration"), f"{kpis['total_duration_hours']} h", "info", "info", "")}
</div>"""
    doors = df[df["door_open"] == True].head(15)
    rows = ""
    for i, (_, row) in enumerate(doors.iterrows(), 1):
        dur = row.get("door_open_duration_sec", 0)
        rows += f"""<tr>
<td>{i}</td>
<td>{_esc(row['timestamp'])}</td>
<td>{row['temperature']:.2f}C</td>
<td>{dur}</td>
<td>{row['latitude']:.4f}, {row['longitude']:.4f}</td>
</tr>"""
    table = f"""<div class="table-card">
<div class="table-card-header">Door Open Events</div>
<table><thead><tr>
<th>Event #</th><th>Timestamp</th><th>Temperature</th><th>Duration (sec)</th><th>Location</th>
</tr></thead><tbody>{rows if rows else '<tr><td colspan="5">No door events recorded</td></tr>'}</tbody></table></div>"""
    c1 = _chart_card(get_t(lang, "gps_route"), charts["gps"], "badge-gps", "GPS").replace('class="chart-card"', 'class="chart-card full-width-chart"', 1)
    return build_alert_banner(kpis, lang) + kpis3 + c1 + table


def build_reports_tab(df, kpis, lang):
    summary_items = [
        (get_t(lang, "shipment"), kpis["shipment_id"]),
        (get_t(lang, "driver"), kpis["driver"]),
        (get_t(lang, "vehicle"), kpis["vehicle"]),
        (get_t(lang, "avg_temp"), f"{kpis['avg_temp']}C"),
        (get_t(lang, "peak_temp"), f"{kpis['max_temp']}C"),
        (get_t(lang, "compliance"), f"{kpis['final_compliance']}%"),
        (get_t(lang, "alerts"), str(kpis["total_alerts"])),
        (get_t(lang, "anomalies_detected"), str(kpis["anomaly_count"])),
        (get_t(lang, "door_events"), str(kpis["total_door_events"])),
        (get_t(lang, "duration"), f"{kpis['total_duration_hours']} h"),
        (get_t(lang, "battery"), f"{kpis['battery_level']}%"),
        (get_t(lang, "signal"), f"{kpis['signal_strength']}%"),
    ]
    summary = ""
    for label, val in summary_items:
        summary += f'<div class="reports-summary-item"><span>{_esc(label)}</span><span>{_esc(val)}</span></div>'
    n = len(df)
    mid_ts = str(df["timestamp"].iloc[n // 2]) if n else kpis["delivery_start"]
    rec_ts = str(df["timestamp"].iloc[3 * n // 4]) if n else kpis["delivery_end"]
    standards = [
        ("WHO GDP", "Good Distribution Practice"),
        ("SCHEDULE M", "Indian Pharmaceutical Standard"),
        ("ICH Q1A", "Stability Guidelines"),
        ("USP 1079", "Good Storage and Distribution Practices"),
    ]
    std_rows = ""
    for name, desc in standards:
        std_rows += f"""<tr>
<td>{_esc(name)}</td><td>{_esc(desc)}</td>
<td><span class="td-badge pass">CERTIFIED</span></td></tr>"""
    milestones = [
        ("Shipment Loaded", kpis["delivery_start"], "Complete"),
        ("Monitoring Started", kpis["delivery_start"], "Complete"),
        ("Peak Excursion", mid_ts, "Recorded"),
        ("Recovery", rec_ts, "Complete"),
        ("Delivery Complete", kpis["delivery_end"], "Complete"),
    ]
    ms_rows = ""
    for name, ts, status in milestones:
        icon = "&#9888;" if status == "Recorded" else "&#9989;"
        ms_rows += f'<div class="timeline-row"><span>{_esc(name)}</span><span>{_esc(ts)}</span><span>{icon} {_esc(status)}</span></div>'
    return f"""<div class="reports-summary"><h3 style="margin-bottom:16px;font-size:15px;">Delivery Summary</h3>
<div class="reports-summary-grid">{summary}</div></div>
<div class="reports-note">Download Report: Run <code>python backend/generate_report.py</code> to generate PDF</div>
<div class="table-card"><div class="table-card-header">Compliance Standards</div>
<table><thead><tr><th>Standard</th><th>Description</th><th>Status</th></tr></thead>
<tbody>{std_rows}</tbody></table></div>
<div class="table-card"><div class="table-card-header">Delivery Timeline</div>
{ms_rows}</div>"""


def build_settings_tab(lang, theme, user):
    checked = "checked" if theme == "dark" else ""
    en_sel = "selected" if lang == "en" else ""
    ta_sel = "selected" if lang == "ta" else ""
    te_sel = "selected" if lang == "te" else ""
    username = user.get("username", "")
    return f"""<div class="settings-grid">
<div class="settings-card">
<div class="settings-card-title">Appearance</div>
<div class="settings-row">
<div><div class="settings-label">{get_t(lang, "dark_mode")} / {get_t(lang, "light_mode")}</div></div>
<label class="toggle-switch"><input type="checkbox" id="themeToggle" {checked} onchange="toggleTheme()"><span class="toggle-slider"></span></label>
</div>
<div class="settings-row">
<div class="settings-label">{get_t(lang, "language")}</div>
<select class="select-input" id="langSelect" onchange="changeLanguage()">
<option value="en" {en_sel}>English</option>
<option value="ta" {ta_sel}>Tamil</option>
<option value="te" {te_sel}>Telugu</option>
</select>
</div>
</div>
<div class="settings-card">
<div class="settings-card-title">Notifications</div>
<div class="settings-row"><div class="settings-label">{get_t(lang, "email_alerts")}</div>
<label class="toggle-switch"><input type="checkbox" checked><span class="toggle-slider"></span></label></div>
<div class="settings-row"><div class="settings-label">{get_t(lang, "telegram_alerts")}</div>
<label class="toggle-switch"><input type="checkbox" checked><span class="toggle-slider"></span></label></div>
<div class="settings-row"><div class="settings-label">SMS Alerts</div>
<label class="toggle-switch"><input type="checkbox"><span class="toggle-slider"></span></label></div>
<div class="settings-row"><div class="settings-label">Push Notifications</div>
<label class="toggle-switch"><input type="checkbox" checked><span class="toggle-slider"></span></label></div>
</div>
<div class="settings-card">
<div class="settings-card-title">{get_t(lang, "general")}</div>
<div class="settings-row"><div class="settings-label">{get_t(lang, "alert_threshold")}</div>
<input class="number-input" type="number" value="{TEMP_MAX}" step="0.1"></div>
<div class="settings-row"><div class="settings-label">Data Refresh Interval</div>
<select class="select-input"><option>30s</option><option>60s</option><option>5min</option></select></div>
<div class="settings-row"><div class="settings-label">Auto Generate Reports</div>
<label class="toggle-switch"><input type="checkbox" checked><span class="toggle-slider"></span></label></div>
<div class="settings-row"><div class="settings-label">Offline Buffering</div>
<label class="toggle-switch"><input type="checkbox" checked><span class="toggle-slider"></span></label></div>
</div>
<div class="settings-card">
<div class="settings-card-title">{get_t(lang, "account")}</div>
<div class="settings-row"><div class="settings-label">{get_t(lang, "username")}</div><div>{_esc(username)}</div></div>
<div class="settings-row"><div class="settings-label">{get_t(lang, "role")}</div><div>{_esc(user.get("role", ""))}</div></div>
<div class="settings-row"><div class="settings-label">{get_t(lang, "version")}</div><div>Cold Chain Monitor v1.0</div></div>
<div class="settings-row"><div class="settings-label">Last Login</div><div>{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div></div>
</div>
</div>
<button class="save-btn" onclick="saveSettings()">Save Settings</button>
<script>
function toggleTheme() {{
  const isDark = document.getElementById('themeToggle').checked;
  const lang = new URLSearchParams(window.location.search).get('lang') || 'en';
  const tab = new URLSearchParams(window.location.search).get('tab') || 'overview';
  window.location.href = `/?tab=${{tab}}&lang=${{lang}}&theme=${{isDark ? 'dark' : 'light'}}`;
}}
function changeLanguage() {{
  const lang = document.getElementById('langSelect').value;
  const theme = new URLSearchParams(window.location.search).get('theme') || 'dark';
  const tab = new URLSearchParams(window.location.search).get('tab') || 'overview';
  window.location.href = `/?tab=${{tab}}&lang=${{lang}}&theme=${{theme}}`;
}}
function saveSettings() {{
  const btn = document.querySelector('.save-btn');
  btn.textContent = 'Settings Saved!';
  btn.style.background = 'linear-gradient(135deg, #2e7d32, #1b5e20)';
  setTimeout(() => {{ btn.textContent = 'Save Settings'; btn.style.background = ''; }}, 2000);
}}
</script>"""


def build_dashboard_page(tab, kpis, charts, user, lang, theme, df):
    user_with_name = dict(user)
    session_user = user_with_name
    title = get_t(lang, TAB_TITLES.get(tab, "overview"))
    if tab == "overview":
        content = build_overview_tab(kpis, charts, lang)
    elif tab == "temperature":
        content = build_temperature_tab(df, kpis, charts, lang)
    elif tab == "anomalies":
        content = build_anomalies_tab(df, kpis, charts, lang)
    elif tab == "gps":
        content = build_gps_tab(df, kpis, charts, lang)
    elif tab == "reports":
        content = build_reports_tab(df, kpis, lang)
    elif tab == "settings":
        content = build_settings_tab(lang, theme, session_user)
    else:
        content = build_overview_tab(kpis, charts, lang)
    return f"""<!DOCTYPE html>
<html lang="{lang}">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ColdChain Monitor - {_esc(title)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>{get_css(theme, lang)}</style>
</head>
<body>
<div class="app-layout">
{build_sidebar(tab, kpis, user, lang, theme)}
<div class="main-content">
{build_header(tab, kpis, user, lang, theme)}
<div class="page-content">
{build_info_strip(kpis, lang)}
{content}
</div>
</div>
</div>
</body>
</html>"""


def build_login_page(error=""):
    err_html = f'<div class="login-error">❌ {_esc(error)}</div>' if error else ""
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>ColdChain Monitor - Login</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>{LOGIN_CSS}</style>
</head>
<body>
<div class="login-page">
<div class="login-container">
<div class="login-left">
<div class="login-brand">
<span class="brand-icon">&#127777;</span>
<div class="brand-name">ColdChain Monitor</div>
<div class="brand-tagline">Enterprise Pharmaceutical<br>Cold Chain Intelligence Platform</div>
</div>
<ul class="login-features">
<li>Real-time IoT temperature monitoring</li>
<li>LSTM anomaly detection and prediction</li>
<li>Instant Telegram and email alerts</li>
<li>WHO/GDP compliant audit reports</li>
<li>GPS route tracking and analytics</li>
</ul>
<div class="login-compliance">
<span class="compliance-badge">WHO GDP</span>
<span class="compliance-badge">SCHEDULE M</span>
<span class="compliance-badge">ICH Q1A</span>
<span class="compliance-badge">USP 1079</span>
</div>
</div>
<div class="login-right">
<h2>Welcome back</h2>
<p class="login-sub">Sign in to your monitoring dashboard</p>
{err_html}
<form method="POST" action="/login">
<div class="form-group">
<label class="form-label">Username</label>
<input class="form-input" type="text" name="username" placeholder="Enter username" required autofocus>
</div>
<div class="form-group">
<label class="form-label">Password</label>
<input class="form-input" type="password" name="password" placeholder="Enter password" required>
</div>
        <button class="login-btn" type="submit">Sign In →</button>
</form>
<div class="demo-hint">
Demo: <span>admin</span> / <span>coldchain123</span>
&nbsp;|&nbsp; <span>driver</span> / <span>driver123</span>
</div>
</div>
</div>
</div>
</body>
</html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        path = parsed.path
        tab = params.get("tab", ["overview"])[0]
        lang = params.get("lang", ["en"])[0]
        theme = params.get("theme", ["dark"])[0]
        if lang not in ("en", "ta", "te"):
            lang = "en"
        if theme not in ("dark", "light"):
            theme = "dark"
        if tab not in ("overview", "temperature", "anomalies", "gps", "reports", "settings"):
            tab = "overview"
        token = get_cookie_token(self)
        session = get_session(token) if token else None
        if path == "/logout":
            if token:
                delete_session(token)
            self.send_response(302)
            self.send_header("Location", "/login")
            self.send_header("Set-Cookie", "session_token=; Max-Age=0; Path=/")
            self.end_headers()
            return
        if path == "/login" or (path == "/" and not session):
            html = build_login_page()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))
            return
        if not session and path != "/login":
            self.send_response(302)
            self.send_header("Location", "/login")
            self.end_headers()
            return
        if path == "/" and session:
            username = session["username"]
            user = dict(USERS[username])
            user["username"] = username
            df = load_data()
            kpis = get_kpis(df)
            charts = {
                "temp": generate_temp_chart(df),
                "spoilage": generate_spoilage_chart(df),
                "humidity": generate_humidity_chart(df),
                "gps": generate_gps_chart(df),
                "reconstruction": generate_reconstruction_chart(df),
                "compliance": generate_compliance_chart(df),
            }
            html = build_dashboard_page(tab, kpis, charts, user, lang, theme, df)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/login":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            params = parse_qs(body)
            username = params.get("username", [""])[0].strip()
            password = params.get("password", [""])[0].strip()
            if username in USERS and USERS[username]["password"] == password:
                token = create_session(username)
                self.send_response(302)
                self.send_header("Location", "/?tab=overview&lang=en&theme=dark")
                self.send_header("Set-Cookie", f"session_token={token}; Path=/; HttpOnly")
                self.end_headers()
            else:
                html = build_login_page(error="Invalid username or password. Please try again.")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(html.encode("utf-8"))

    def log_message(self, format, *args):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {self.path} — served")


if __name__ == "__main__":
    HOST = "localhost"
    PORT = 8000
    print("=" * 60)
    print("  🌡️  COLDCHAIN MONITOR — ENTERPRISE DASHBOARD")
    print("=" * 60)
    print(f"  URL     : http://{HOST}:{PORT}")
    print("  Login   : admin / coldchain123")
    print("  Login   : driver / driver123")
    print("=" * 60)
    print("  Tabs    : Overview, Temperature, Anomalies, GPS, Reports, Settings")
    print("  Langs   : English, Tamil, Telugu")
    print("  Themes  : Dark, Light")
    print("=" * 60)
    print("  Press Ctrl+C to stop.")
    print("=" * 60)
    server = HTTPServer((HOST, PORT), DashboardHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")

