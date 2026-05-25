/*
  Cold Chain Monitor - ESP32 Firmware
  Sensors: DS18B20 (temp), SHT31 (humidity), NEO-6M (GPS), Reed Switch (door)
  Connectivity: LTE-M -> MQTT -> AWS IoT Core
  Data interval: 30 seconds
  This file is the hardware implementation reference.
  For simulation, see data/simulate_data.py
*/

#include <Arduino.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <Wire.h>
#include "Adafruit_SHT31.h"

#define ONE_WIRE_BUS 4
#define DOOR_PIN 13
#define SEND_INTERVAL 30000

OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);
Adafruit_SHT31 sht31 = Adafruit_SHT31();

const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";
const char* mqtt_server = "broker.hivemq.com";
const char* topic = "coldchain/shipment";

WiFiClient espClient;
PubSubClient client(espClient);

void setup() {
  Serial.begin(115200);
  sensors.begin();
  sht31.begin(0x44);
  pinMode(DOOR_PIN, INPUT_PULLUP);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) delay(500);
  client.setServer(mqtt_server, 1883);
}

void loop() {
  sensors.requestTemperatures();
  float temp = sensors.getTempCByIndex(0);
  float humidity = sht31.readHumidity();
  bool doorOpen = digitalRead(DOOR_PIN) == HIGH;

  String payload = "{";
  payload += "\"temperature\":" + String(temp) + ",";
  payload += "\"humidity\":" + String(humidity) + ",";
  payload += "\"door_open\":" + String(doorOpen ? "true" : "false") + ",";
  payload += "\"shipment_id\":\"SHIPMENT_1234\"";
  payload += "}";

  if (client.connected()) {
    client.publish(topic, payload.c_str());
  }
  delay(SEND_INTERVAL);
}
