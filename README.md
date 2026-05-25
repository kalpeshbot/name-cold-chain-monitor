# 🌡️ Cold Chain Monitor

> **Real-time IoT + ML system that predicts pharmaceutical temperature excursions before they happen.**

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange?style=flat-square&logo=tensorflow)
![MQTT](https://img.shields.io/badge/MQTT-HiveMQ-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

---

## 🎯 Problem Statement

Pharmaceutical drugs like vaccines, insulin, and blood samples require strict temperature control (2°C – 8°C) during transport. A single temperature excursion can destroy an entire shipment worth lakhs of rupees.

**Current systems only DETECT problems AFTER they happen — too late.**

---

## 💡 Solution

An end-to-end IoT + ML pipeline that:
- 📡 Monitors drug shipments every 30 seconds via ESP32 sensors
- 🧠 Predicts temperature excursions **10-15 minutes before** they happen
- 🚨 Sends instant Telegram alerts with GPS location and spoilage probability
- 📄 Auto-generates WHO/GDP compliant PDF audit reports per delivery
- 📊 Renders a full matplotlib monitoring dashboard

---

## 🏗️ System Architecture
ESP32 (DS18B20 + SHT31 + GPS + Reed Switch)
↓ MQTT every 30 seconds
AWS IoT Core / HiveMQ Broker
↓ streams to
Python MQTT Subscriber (mqtt_subscriber.py)
↓ feeds into
LSTM Autoencoder (train_lstm.py + detect_anomaly.py)
↓ triggers
Telegram Alert Bot (alert_bot.py)
↓ generates
PDF Audit Report (generate_report.py)
↓ visible on
Matplotlib Dashboard (visualize.py)

---

## 📁 Project Structure
cold-chain-monitor/
├── config.py
├── requirements.txt
├── firmware/
│   └── esp32_cold_chain.ino
├── data/
│   ├── simulate_data.py
│   └── sample_data.csv
├── ml/
│   ├── train_lstm.py
│   ├── detect_anomaly.py
│   └── model/
│       ├── lstm_model.keras
│       ├── scaler.pkl
│       └── threshold.pkl
├── backend/
│   ├── mqtt_subscriber.py
│   ├── alert_bot.py
│   └── generate_report.py
├── dashboard/
│   └── visualize.py
├── reports/
│   ├── SHIPMENT_1234_report.pdf
│   └── dashboard.png
└── logs/
├── mqtt_log.txt
└── alert_log.txt

---

## 🚀 Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/kalpeshbot/cold-chain-monitor-
cd cold-chain-monitor-
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure credentials
Open config.py and set:
```python
TELEGRAM_BOT_TOKEN = "your_bot_token_here"
TELEGRAM_CHAT_ID   = "your_chat_id_here"
```

### 4. Generate simulated sensor data
```bash
python data/simulate_data.py
```

### 5. Train the LSTM model
```bash
python ml/train_lstm.py
```

### 6. Run anomaly detection
```bash
python ml/detect_anomaly.py
```

### 7. Simulate live MQTT stream
```bash
python backend/mqtt_subscriber.py
```

### 8. Send Telegram alerts
```bash
python backend/alert_bot.py
```

### 9. Generate PDF audit report
```bash
python backend/generate_report.py
```

### 10. Launch dashboard
```bash
python dashboard/visualize.py
```

---

## 🧠 ML Model — LSTM Autoencoder

| Component | Detail |
|-----------|--------|
| Architecture | LSTM Autoencoder (Encoder-Decoder) |
| Input | 20 timestep sequences (temperature, humidity, spoilage) |
| Encoder | LSTM(64) → LSTM(32) → RepeatVector(20) |
| Decoder | LSTM(32) → LSTM(64) → TimeDistributed(Dense(3)) |
| Loss | Mean Absolute Error (MAE) |
| Training data | Normal readings only (2°C – 8°C) |
| Anomaly trigger | Reconstruction error > learned threshold |
| Prediction window | 10–15 minutes ahead |

---

## 📊 Dashboard Panels

| Panel | Description |
|-------|-------------|
| KPI Header | Avg temp, peak temp, compliance %, alerts, spoilage risk |
| Temperature Timeline | Normal vs anomaly vs alert zones with safe band |
| Spoilage Probability | ML-derived risk over time with warning/critical lines |
| Humidity Timeline | With door open event markers |
| Reconstruction Error | LSTM MAE with anomaly threshold line |
| GPS Route Map | Color-coded by temperature with alert zones |
| Compliance Score | Over time with pass/fail threshold |
| Delivery Summary | Full stats table with color-coded status |

---

## 📄 PDF Report Sections

1. Shipment Information
2. Compliance Summary (PASS / FAIL)
3. Temperature Summary
4. Humidity and Door Events
5. Anomaly Detection Results (top 10)
6. Digital Signature and Audit Hash

---

## 🌡️ Compliance Standards Met

| Standard | Description |
|----------|-------------|
| WHO GDP | Good Distribution Practice |
| SCHEDULE M | Indian Pharmaceutical Standard |
| ICH Q1A | Stability Guidelines |
| USP 1079 | Good Storage and Distribution Practices |

---

## 💰 Real World Impact

| Metric | Value |
|--------|-------|
| Indian cold chain market | $700M+ |
| Shipments lost to excursions | ~20% |
| Estimated loss reduction | 15–20% |
| Savings per truck per year | ₹50,000 – ₹2,00,000 |

---

## 🛠️ Hardware Reference

| Component | Purpose |
|-----------|---------|
| ESP32 | Main microcontroller |
| DS18B20 | Temperature sensor |
| SHT31 | Humidity sensor |
| NEO-6M | GPS tracking |
| Reed Switch | Door open/close detection |
| LTE-M Module | Cellular connectivity |

Hardware implementation is in firmware/esp32_cold_chain.ino
For software-only demo, all hardware is simulated via data/simulate_data.py

---

## 📦 Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| tensorflow | 2.16.1 | LSTM Autoencoder |
| pandas | 2.2.2 | Data processing |
| numpy | 1.26.4 | Numerical computing |
| matplotlib | 3.8.4 | Dashboard rendering |
| fpdf2 | 2.7.9 | PDF report generation |
| scikit-learn | 1.4.2 | Data scaling |
| requests | 2.31.0 | Telegram API |
| paho-mqtt | 2.0.0 | MQTT protocol |
| python-dotenv | 1.0.1 | Environment config |

---

## 📜 License

MIT License — free to use, modify, and distribute.

---

*Cold Chain Monitor v1.0 — Protecting lives through intelligent temperature monitoring.*
