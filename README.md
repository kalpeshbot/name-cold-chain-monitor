# Cold Chain Compliance Monitor

**End-to-end IoT and ML pipeline for pharmaceutical drug shipment temperature integrity monitoring.**

---

## Overview

The Cold Chain Compliance Monitor is a production-grade embedded system designed to ensure the integrity of temperature-sensitive pharmaceutical shipments during transit. The system combines hardware sensor nodes, real-time ML anomaly detection, and automated WHO/GDP-compliant audit report generation into a single unified pipeline.

Unlike traditional monitoring systems that detect excursions after the fact, this system predicts temperature violations before they occur — enabling proactive intervention and preventing spoilage.

---

## Problem Statement

Pharmaceutical drugs including vaccines, insulin, and biological samples require strict temperature control between 2 degrees Celsius and 8 degrees Celsius during transport. A single undetected temperature excursion can render an entire shipment unusable, resulting in significant financial loss and potential patient safety risks. Existing systems are reactive — they log data but do not predict or prevent excursions in real time.

---

## Solution Architecture

```
ESP32 Sensor Node
      |
      | (MQTT over LTE-M)
      v
AWS IoT Core
      |
      v
MQTT Subscriber (Python)
      |
      |--- InfluxDB (Time Series Storage)
      |
      v
LSTM Autoencoder (Anomaly Detection)
      |
      |--- Telegram Bot (Real-time Alerts)
      |--- PDF Report Generator (WHO/GDP Compliance)
      v
Grafana Dashboard (Live Visualization)
```

---

## Features

- Real-time temperature and humidity monitoring via DS18B20 and SHT31 sensors
- GPS-based location tracking correlated with temperature events
- Door open and close event detection via reed switch
- LSTM Autoencoder anomaly detection running on incoming sensor data
- Predictive excursion alerts sent 10 to 15 minutes before threshold breach
- Spoilage probability score calculated per shipment
- Offline buffering with automatic data replay on network reconnect
- Auto-generated WHO/GDP-compliant PDF audit reports per delivery
- QR code per shipment linking to full audit trail
- Telegram Bot alerts with location, temperature, and recommended action
- Grafana dashboard with live temperature graphs and GPS route maps

---

## Hardware Components

| Component | Purpose |
|-----------|---------|
| ESP32 | Main microcontroller |
| DS18B20 | Temperature sensing |
| SHT31 | Humidity sensing |
| NEO-6M GPS Module | Location tracking |
| Reed Switch | Door open/close detection |
| LTE-M Module | Cellular connectivity |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Firmware | C++, Arduino IDE, FreeRTOS |
| Protocol | MQTT |
| Cloud | AWS IoT Core |
| Streaming | Apache Kafka |
| Database | InfluxDB, TimescaleDB |
| ML | Python, TensorFlow, LSTM Autoencoder |
| Backend | FastAPI, Docker |
| Alerts | Telegram Bot API |
| Reports | ReportLab (PDF) |
| Dashboard | Grafana |

---

## Project Structure

```
cold-chain-monitor/
├── firmware/
│   └── esp32_cold_chain.ino       # ESP32 sensor + MQTT firmware
├── ml/
│   ├── train_lstm.py              # LSTM Autoencoder training
│   ├── detect_anomaly.py          # Real-time anomaly detection
│   └── model/                     # Saved model weights
├── backend/
│   ├── mqtt_subscriber.py         # MQTT data ingestion
│   ├── alert_bot.py               # Telegram alert system
│   └── generate_report.py         # WHO/GDP PDF report generator
├── dashboard/
│   └── visualize.py               # Grafana / matplotlib dashboard
├── data/
│   └── sample_data.csv            # Sample sensor data for testing
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.11 or above
- Arduino IDE 2.0
- Docker Desktop
- AWS account with IoT Core enabled
- Telegram Bot token

### Installation

Clone the repository:

```bash
git clone https://github.com/kalpeshbuilds/cold-chain-monitor.git
cd cold-chain-monitor
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Flash the ESP32 firmware:

1. Open `firmware/esp32_cold_chain.ino` in Arduino IDE
2. Configure your WiFi credentials and MQTT broker endpoint in the config section
3. Flash to ESP32 via USB

### Running the Backend

```bash
python backend/mqtt_subscriber.py
```

### Running Anomaly Detection

```bash
python ml/detect_anomaly.py
```

### Running the Alert Bot

```bash
python backend/alert_bot.py
```

---

## ML Model

The anomaly detection engine uses an LSTM Autoencoder trained on historical normal temperature data (2 degrees Celsius to 8 degrees Celsius range). The model learns the expected patterns of temperature and humidity over time. During inference, reconstruction error is calculated on incoming sensor windows — a high reconstruction error indicates an anomaly.

Excursion prediction works by fitting a trend line over the last 10 sensor readings and projecting the temperature value 10 to 15 minutes ahead. If the projected value exceeds the threshold, an alert is triggered before the actual excursion occurs.

---

## Compliance Standards

This system is designed with reference to the following regulatory standards:

- WHO Good Distribution Practice (GDP)
- Schedule M — Indian Pharmaceutical Standards
- ICH Q1A Stability Guidelines
- USP 1079 Good Storage and Shipping Practices

---

## Sample Alert Format

```
COLD CHAIN ALERT — Shipment #1234

Status   : Temperature Rising
Current  : 7.4 degrees Celsius
Predicted: 8.3 degrees Celsius in 12 minutes
Humidity : 68%
Location : NH-44, Km 234, Andhra Pradesh
Door     : Opened 3 times in last 30 minutes

Recommended Action: Inspect vehicle cooling unit immediately.
```

---

## Future Improvements

- Multi-node support for fleet-level monitoring
- Weather API integration for route risk scoring
- Driver behavior scoring based on door open patterns and vehicle speed
- Mobile application for logistics managers
- Blockchain-based immutable audit trail

---

## Author

**Kalpesh V L**
IoT Engineer and ML Systems Builder
VIT-AP University, CSE 2028
linkedin.com/in/kalpeshbuilds

---

## License

This project is licensed under the MIT License.
