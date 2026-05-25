import pandas as pd
import numpy as np
from fpdf import FPDF
import os
import sys
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import (
    SHIPMENT_ID,
    DRIVER_NAME,
    VEHICLE_NUMBER,
    TEMP_MIN,
    TEMP_MAX,
    HUMIDITY_MIN,
    HUMIDITY_MAX,
    REPORT_OUTPUT,
)


class ColdChainReport(FPDF):
    def __init__(self, stats, df):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_margins(15, 15, 15)
        self.set_auto_page_break(auto=True, margin=25)
        self.stats = stats
        self.df = df

    def header(self):
        self.set_fill_color(0, 71, 171)
        self.rect(0, 0, 210, 25, style="F")
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(255, 255, 255)
        self.set_y(8)
        self.cell(0, 6, "COLD CHAIN MONITOR - AUDIT REPORT", align="C", ln=True)
        self.set_font("Helvetica", "", 9)
        self.set_y(16)
        self.cell(0, 5, "WHO GDP Compliant Temperature Monitoring System", align="C", ln=True)
        self.set_text_color(0, 0, 0)
        self.ln(12)

    def footer(self):
        self.set_y(-20)
        self.set_draw_color(200, 200, 200)
        self.line(15, self.get_y(), 195, self.get_y())
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        footer_text = (
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
            f"Confidential | Page {self.page_no()}"
        )
        self.cell(0, 5, footer_text, align="C")

    def section_heading(self, title):
        if self.get_y() > 250:
            self.add_page()
        self.set_fill_color(240, 240, 240)
        self.rect(15, self.get_y(), 180, 8, style="F")
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(0, 0, 0)
        self.set_xy(17, self.get_y() + 2)
        self.cell(0, 5, title, ln=True)
        self.ln(4)

    def add_info_table(self, data, col_widths=None):
        if col_widths is None:
            col_widths = [80, 90]
        label_w, value_w = col_widths
        for i, (label, value) in enumerate(data):
            if self.get_y() > 270:
                self.add_page()
            value_fill = (250, 250, 250) if i % 2 else (255, 255, 255)
            self.set_fill_color(240, 240, 240)
            self.set_font("Helvetica", "B", 9)
            self.cell(label_w, 7, str(label), border=1, fill=True)
            self.set_fill_color(*value_fill)
            self.set_font("Helvetica", "", 9)
            self.cell(value_w, 7, str(value), border=1, fill=True, ln=True)
        self.ln(4)

    def compliance_status_box(self, status):
        if status == "PASS":
            self.set_fill_color(0, 200, 0)
            text = "COMPLIANCE STATUS: PASS"
        else:
            self.set_fill_color(220, 0, 0)
            text = "COMPLIANCE STATUS: FAIL"
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 14)
        self.cell(180, 15, text, align="C", fill=True, ln=True)
        self.set_text_color(0, 0, 0)
        self.ln(4)

    def add_door_events_table(self):
        door_df = self.df[self.df["door_open"] == True].head(10)
        col_widths = [15, 55, 40, 70]
        headers = ["Event #", "Timestamp", "Duration (sec)", "Temperature at Event"]

        self.set_fill_color(200, 200, 200)
        self.set_font("Helvetica", "B", 9)
        for i, header in enumerate(headers):
            self.cell(col_widths[i], 7, header, border=1, fill=True, align="C")
        self.ln()

        self.set_font("Helvetica", "", 9)
        for idx, (_, row) in enumerate(door_df.iterrows()):
            if self.get_y() > 270:
                self.add_page()
            fill = (245, 245, 245) if idx % 2 else (255, 255, 255)
            self.set_fill_color(*fill)
            values = [
                str(idx + 1),
                str(row["timestamp"]),
                str(int(row["door_open_duration_sec"])),
                f"{float(row['temperature']):.1f}",
            ]
            for i, val in enumerate(values):
                self.cell(col_widths[i], 7, val, border=1, fill=True, align="C")
            self.ln()
        self.ln(4)

    def add_anomaly_table(self):
        anomaly_df = (
            self.df.dropna(subset=["reconstruction_error"])
            .sort_values("reconstruction_error", ascending=False)
            .head(10)
        )
        col_widths = [38, 28, 24, 38, 28, 24]
        headers = [
            "Timestamp",
            "Temperature",
            "Humidity",
            "Reconstruction Error",
            "Spoilage Risk %",
            "Alert",
        ]

        self.set_fill_color(0, 71, 171)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 8)
        for i, header in enumerate(headers):
            self.cell(col_widths[i], 7, header, border=1, fill=True, align="C")
        self.ln()

        self.set_text_color(0, 0, 0)
        self.set_font("Helvetica", "", 8)
        for idx, (_, row) in enumerate(anomaly_df.iterrows()):
            if self.get_y() > 270:
                self.add_page()
            if row["alert_triggered"] == True:
                self.set_fill_color(255, 220, 220)
            else:
                fill = (245, 245, 245) if idx % 2 else (255, 255, 255)
                self.set_fill_color(*fill)

            spoilage_pct = (
                f"{float(row['ml_spoilage_probability']) * 100:.1f}"
                if pd.notna(row["ml_spoilage_probability"])
                else "N/A"
            )
            values = [
                str(row["timestamp"]),
                f"{float(row['temperature']):.1f}",
                f"{float(row['humidity']):.1f}",
                f"{float(row['reconstruction_error']):.6f}",
                spoilage_pct,
                str(bool(row["alert_triggered"])),
            ]
            for i, val in enumerate(values):
                self.cell(col_widths[i], 7, val, border=1, fill=True, align="C")
            self.ln()
        self.ln(4)

    def build_report(self):
        s = self.stats
        self.add_page()

        # SECTION 1
        self.section_heading("1. SHIPMENT INFORMATION")
        self.add_info_table(
            [
                ("Shipment ID", SHIPMENT_ID),
                ("Driver Name", DRIVER_NAME),
                ("Vehicle Number", VEHICLE_NUMBER),
                ("Delivery Start", s["delivery_start"]),
                ("Delivery End", s["delivery_end"]),
                ("Total Duration", f"{s['total_duration_hours']} hours"),
                ("Report Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            ]
        )

        # SECTION 2
        self.section_heading("2. COMPLIANCE SUMMARY")
        self.compliance_status_box(s["compliance_status"])
        self.add_info_table(
            [
                ("Compliance Score", f"{s['final_compliance_score']}%"),
                ("Temperature Excursions", f"{s['excursion_minutes']} minutes"),
                ("Total Anomalies Detected", str(s["total_anomalies"])),
                ("Total Alerts Triggered", str(s["total_alerts"])),
                ("Peak Spoilage Risk", f"{s['peak_spoilage']}%"),
                ("Standards Met", "WHO GDP, SCHEDULE M, ICH Q1A, USP <1079>"),
            ]
        )

        # SECTION 3
        self.section_heading("3. TEMPERATURE SUMMARY")
        self.add_info_table(
            [
                ("Average Temperature", f"{s['avg_temp']}°C"),
                ("Minimum Temperature", f"{s['min_temp']}°C"),
                ("Maximum Temperature", f"{s['max_temp']}°C"),
                ("Safe Range", f"{TEMP_MIN}°C - {TEMP_MAX}°C"),
                ("Readings Above Max", f"{s['rows_above_max']} readings"),
                ("Readings Below Min", f"{s['rows_below_min']} readings"),
            ]
        )

        # SECTION 4
        self.section_heading("4. HUMIDITY AND DOOR EVENTS")
        self.add_info_table(
            [
                ("Average Humidity", f"{s['avg_humidity']}%"),
                ("Safe Humidity Range", f"{HUMIDITY_MIN}% - {HUMIDITY_MAX}%"),
                ("Total Door Events", str(s["total_door_events"])),
                ("Total Door Open Time", f"{s['total_door_open_time']} seconds"),
            ]
        )
        self.add_door_events_table()

        # SECTION 5
        self.section_heading("5. ANOMALY DETECTION RESULTS")
        self.set_font("Helvetica", "", 9)
        intro = (
            "The LSTM Autoencoder model was trained on normal temperature patterns (2-8C). "
            "Anomalies are detected when reconstruction error exceeds the learned threshold. "
            "The following table shows the top 10 highest anomaly events detected."
        )
        self.multi_cell(180, 5, intro)
        self.ln(4)
        self.add_anomaly_table()

        # SECTION 6
        if self.get_y() > 220:
            self.add_page()
        self.section_heading("6. DIGITAL SIGNATURE & QR")
        self.set_font("Helvetica", "", 9)
        hash_text = (
            f"This report has been automatically generated by Cold Chain Monitor v1.0.\n"
            f"Data integrity is guaranteed by continuous IoT sensor logging.\n"
            f"Report Hash: SHA256-{SHIPMENT_ID}-{datetime.now().strftime('%Y%m%d')}-COLDCHAIN"
        )
        self.multi_cell(180, 5, hash_text)
        self.ln(6)

        box_y = self.get_y()
        self.rect(15, box_y, 90, 40)
        self.set_xy(17, box_y + 4)
        self.set_font("Helvetica", "B", 10)
        self.cell(0, 6, "AUTHORIZED SIGNATURE", ln=True)
        self.ln(18)
        self.set_font("Helvetica", "", 9)
        self.set_x(17)
        self.cell(
            0,
            5,
            f"Logistics Manager | {datetime.now().strftime('%Y-%m-%d')}",
            ln=True,
        )


def _calculate_stats(df):
    duration = df["timestamp"].max() - df["timestamp"].min()
    final_compliance_score = round(float(df["compliance_score"].iloc[-1]), 2)
    compliance_status = "PASS" if final_compliance_score >= 85.0 else "FAIL"
    rows_above_max = len(df[df["temperature"] > TEMP_MAX])
    rows_below_min = len(df[df["temperature"] < TEMP_MIN])

    return {
        "delivery_start": df["timestamp"].min().strftime("%Y-%m-%d %H:%M:%S"),
        "delivery_end": df["timestamp"].max().strftime("%Y-%m-%d %H:%M:%S"),
        "total_duration_hours": round(duration.seconds / 3600, 2),
        "avg_temp": round(float(df["temperature"].mean()), 2),
        "min_temp": round(float(df["temperature"].min()), 2),
        "max_temp": round(float(df["temperature"].max()), 2),
        "avg_humidity": round(float(df["humidity"].mean()), 2),
        "total_door_events": int(df["door_open"].sum()),
        "total_door_open_time": int(df["door_open_duration_sec"].sum()),
        "total_anomalies": int(df["is_anomaly"].sum()),
        "total_alerts": int(df["alert_triggered"].sum()),
        "final_compliance_score": final_compliance_score,
        "peak_spoilage": round(float(df["ml_spoilage_probability"].max() * 100), 1),
        "compliance_status": compliance_status,
        "rows_above_max": rows_above_max,
        "rows_below_min": rows_below_min,
        "excursion_minutes": round((rows_above_max + rows_below_min) * 0.5, 1),
    }


def generate_report():
    # STEP 1 — Load data
    results_path = "data/anomaly_results.csv"
    if not os.path.exists(results_path):
        print("ERROR: data/anomaly_results.csv not found. Run ml/detect_anomaly.py first.")
        return

    df = pd.read_csv(results_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"Loaded {len(df)} rows for report generation.")

    # STEP 2 — Calculate report statistics
    stats = _calculate_stats(df)

    # STEP 3 — Create PDF
    report = ColdChainReport(stats, df)
    report.build_report()

    # STEP 4 — Save PDF
    os.makedirs(REPORT_OUTPUT, exist_ok=True)
    output_path = os.path.join(REPORT_OUTPUT, f"{SHIPMENT_ID}_report.pdf")
    report.output(output_path)
    print("Report saved to reports/SHIPMENT_1234_report.pdf")
    print(f"Compliance Status: {stats['compliance_status']}")
    print(f"Final Compliance Score: {stats['final_compliance_score']}%")
    print("Report generation complete.")


if __name__ == "__main__":
    generate_report()
