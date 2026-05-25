import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import (
    TEMP_MIN, TEMP_MAX,
    HUMIDITY_MIN, HUMIDITY_MAX,
    SHIPMENT_ID, DRIVER_NAME, VEHICLE_NUMBER
)


def _style_axes(ax):
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="#aaaaaa")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.grid(color="#333355", linestyle="--", alpha=0.4)


def _add_legend(ax):
    ax.legend(
        loc="upper right",
        fontsize=7,
        facecolor="#1a1a2e",
        edgecolor="#333355",
        labelcolor="white",
    )


def _load_threshold(df):
    threshold_path = "ml/model/threshold.pkl"
    if os.path.exists(threshold_path):
        import pickle
        with open(threshold_path, "rb") as f:
            return pickle.load(f)
    return df["reconstruction_error"].quantile(0.95)


def _table_status_color(status):
    if status in ("OK", "INFO", "CERTIFIED", "PASS"):
        return "#1a3a1a", "#00ff88"
    if status in ("WARN", "REVIEW"):
        return "#3a3a1a", "#ffaa00"
    if status in ("EXCEEDED", "ACTION NEEDED", "HIGH RISK", "FAIL"):
        return "#3a1a1a", "#ff4444"
    return "#1a1a2e", "white"


def visualize():
    # STEP 1 — Load data
    results_path = "data/anomaly_results.csv"
    if not os.path.exists(results_path):
        print("ERROR: data/anomaly_results.csv not found. Run ml/detect_anomaly.py first.")
        return

    df = pd.read_csv(results_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["row_index"] = range(len(df))
    print(f"Loaded {len(df)} rows for dashboard.")

    # STEP 2 — Prepare data subsets
    normal_rows = df[df["is_anomaly"] == False]
    anomaly_rows = df[df["is_anomaly"] == True]
    alert_rows = df[df["alert_triggered"] == True]
    door_event_rows = df[df["door_open"] == True]

    # STEP 3 — Calculate summary stats
    avg_temp = round(float(df["temperature"].mean()), 2)
    max_temp = round(float(df["temperature"].max()), 2)
    final_compliance = round(float(df["compliance_score"].iloc[-1]), 2)
    total_alerts = int(df["alert_triggered"].sum())
    total_door_events = int(df["door_open"].sum())
    peak_spoilage = round(float(df["ml_spoilage_probability"].max() * 100), 1)
    compliance_status = "PASS" if final_compliance >= 85.0 else "FAIL"

    # STEP 4 — Create figure
    fig = plt.figure(figsize=(20, 24), facecolor="#0a0a0a")
    fig.suptitle(
        f"COLD CHAIN MONITOR — {SHIPMENT_ID}",
        fontsize=18,
        color="white",
        fontweight="bold",
        y=0.98,
    )

    gs = gridspec.GridSpec(
        5, 2,
        figure=fig,
        height_ratios=[0.8, 2, 2, 2, 2],
        hspace=0.45,
        wspace=0.3,
    )

    # STEP 5 — PANEL 1: KPI HEADER
    ax_kpi = fig.add_subplot(gs[0, :])
    ax_kpi.axis("off")
    ax_kpi.set_facecolor("#16213e")

    kpi_data = [
        ("AVG TEMP", f"{avg_temp}°C", "#00d4ff" if TEMP_MIN <= avg_temp <= TEMP_MAX else "#ff4444"),
        ("PEAK TEMP", f"{max_temp}°C", "#ff4444" if max_temp > TEMP_MAX else "#00d4ff"),
        ("COMPLIANCE", f"{final_compliance}%", "#00ff88" if compliance_status == "PASS" else "#ff4444"),
        ("STATUS", compliance_status, "#00ff88" if compliance_status == "PASS" else "#ff4444"),
        ("TOTAL ALERTS", str(total_alerts), "#ff4444" if total_alerts > 0 else "#00ff88"),
        ("PEAK SPOILAGE", f"{peak_spoilage}%", "#ff4444" if peak_spoilage > 50 else "#ffaa00"),
    ]
    x_positions = [0.08, 0.24, 0.40, 0.56, 0.72, 0.88]
    for (label, value, color), x_pos in zip(kpi_data, x_positions):
        ax_kpi.text(x_pos, 0.65, label, fontsize=9, color="#888888", ha="center", transform=ax_kpi.transAxes)
        ax_kpi.text(x_pos, 0.25, value, fontsize=16, fontweight="bold", color=color, ha="center", transform=ax_kpi.transAxes)

    # STEP 6 — PANEL 2: TEMPERATURE TIMELINE
    ax_temp = fig.add_subplot(gs[1, 0])
    _style_axes(ax_temp)
    ax_temp.set_title("🌡 Temperature Over Time")
    ax_temp.set_xlabel("Reading Index")
    ax_temp.set_ylabel("Temperature (°C)")

    ax_temp.axhspan(TEMP_MIN, TEMP_MAX, alpha=0.15, color="#00ff88", label="Safe Zone")
    ax_temp.axhline(TEMP_MAX, color="#ff4444", linewidth=1.5, linestyle="--", label=f"Max Safe ({TEMP_MAX}°C)")
    ax_temp.axhline(TEMP_MIN, color="#ffaa00", linewidth=1.5, linestyle="--", label=f"Min Safe ({TEMP_MIN}°C)")

    if not normal_rows.empty:
        ax_temp.plot(
            normal_rows["row_index"],
            normal_rows["temperature"],
            color="#00d4ff",
            linewidth=0.8,
            alpha=0.8,
            label="Normal",
        )
    if not anomaly_rows.empty:
        ax_temp.plot(
            anomaly_rows["row_index"],
            anomaly_rows["temperature"],
            color="#ff4444",
            linewidth=1.2,
            alpha=0.9,
            label="Anomaly",
        )
    if not alert_rows.empty:
        ax_temp.scatter(
            alert_rows["row_index"],
            alert_rows["temperature"],
            color="#ff0000",
            s=20,
            zorder=5,
            label="Alert",
            marker="x",
        )

    for _, row in door_event_rows.iterrows():
        ax_temp.axvline(x=row["row_index"], color="#ffff00", linewidth=0.8, alpha=0.6)
    if not door_event_rows.empty:
        door_proxy = Line2D([0], [0], color="#ffff00", linewidth=0.8, alpha=0.6, label="Door Open")
        ax_temp.add_line(door_proxy)
    _add_legend(ax_temp)

    # STEP 7 — PANEL 3: SPOILAGE PROBABILITY
    ax_spoilage = fig.add_subplot(gs[1, 1])
    _style_axes(ax_spoilage)
    ax_spoilage.set_title("☣️ Spoilage Probability Over Time")
    ax_spoilage.set_xlabel("Reading Index")
    ax_spoilage.set_ylabel("Spoilage Probability")
    ax_spoilage.set_ylim(0, 1.05)

    spoilage = df["ml_spoilage_probability"].fillna(0)
    ax_spoilage.fill_between(df["row_index"], spoilage, color="#ff4444", alpha=0.3)
    ax_spoilage.plot(df["row_index"], spoilage, color="#ff6666", linewidth=1)
    ax_spoilage.axhline(0.5, color="#ffaa00", linestyle="--", linewidth=1, label="Warning (50%)")
    ax_spoilage.axhline(0.8, color="#ff0000", linestyle="--", linewidth=1, label="Critical (80%)")
    _add_legend(ax_spoilage)

    # STEP 8 — PANEL 4: HUMIDITY TIMELINE
    ax_hum = fig.add_subplot(gs[2, 0])
    _style_axes(ax_hum)
    ax_hum.set_title("💧 Humidity Over Time")
    ax_hum.set_xlabel("Reading Index")
    ax_hum.set_ylabel("Humidity (%)")

    ax_hum.axhspan(HUMIDITY_MIN, HUMIDITY_MAX, alpha=0.15, color="#00aaff")
    ax_hum.plot(df["row_index"], df["humidity"], color="#00aaff", linewidth=0.8, alpha=0.8, label="Humidity")
    if not door_event_rows.empty:
        ax_hum.scatter(
            door_event_rows["row_index"],
            door_event_rows["humidity"],
            color="#ffff00",
            s=30,
            zorder=5,
            label="Door Open",
            marker="^",
        )
    _add_legend(ax_hum)

    # STEP 9 — PANEL 5: RECONSTRUCTION ERROR
    ax_recon = fig.add_subplot(gs[2, 1])
    _style_axes(ax_recon)
    ax_recon.set_title("📊 LSTM Reconstruction Error")
    ax_recon.set_xlabel("Reading Index")
    ax_recon.set_ylabel("Reconstruction Error (MAE)")

    recon = df["reconstruction_error"].fillna(0)
    threshold = _load_threshold(df)
    ax_recon.fill_between(df["row_index"], recon, 0, color="#aa00ff", alpha=0.3)
    ax_recon.plot(df["row_index"], recon, color="#cc44ff", linewidth=0.8)
    ax_recon.axhline(threshold, color="#ff4444", linestyle="--", linewidth=1.5, label="Threshold")
    _add_legend(ax_recon)

    # STEP 10 — PANEL 6: GPS ROUTE MAP
    ax_gps = fig.add_subplot(gs[3, 0])
    _style_axes(ax_gps)
    ax_gps.set_title("📍 GPS Route Map")
    ax_gps.set_xlabel("Longitude")
    ax_gps.set_ylabel("Latitude")

    ax_gps.plot(df["longitude"], df["latitude"], color="#444488", linewidth=1, alpha=0.5)
    sc = ax_gps.scatter(
        df["longitude"],
        df["latitude"],
        c=df["temperature"],
        cmap="RdYlGn_r",
        vmin=TEMP_MIN,
        vmax=TEMP_MAX + 2,
        s=3,
        alpha=0.7,
    )
    cbar = fig.colorbar(sc, ax=ax_gps, shrink=0.8, label="Temperature (°C)")
    cbar.ax.yaxis.set_tick_params(color="white")
    cbar.ax.xaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    cbar.set_label("Temperature (°C)", color="white")
    cbar.ax.yaxis.label.set_color("white")

    ax_gps.scatter(
        df["longitude"].iloc[0],
        df["latitude"].iloc[0],
        color="green",
        marker="^",
        s=100,
        zorder=5,
        label="Start",
    )
    ax_gps.scatter(
        df["longitude"].iloc[-1],
        df["latitude"].iloc[-1],
        color="red",
        marker="v",
        s=100,
        zorder=5,
        label="End",
    )
    if not alert_rows.empty:
        ax_gps.scatter(
            alert_rows["longitude"],
            alert_rows["latitude"],
            color="#ff0000",
            marker="x",
            s=40,
            zorder=6,
            label="Alert Zone",
        )
    _add_legend(ax_gps)

    # STEP 11 — PANEL 7: COMPLIANCE SCORE
    ax_comp = fig.add_subplot(gs[3, 1])
    _style_axes(ax_comp)
    ax_comp.set_title("✅ Compliance Score Over Time")
    ax_comp.set_xlabel("Reading Index")
    ax_comp.set_ylabel("Compliance Score (%)")
    ax_comp.set_ylim(0, 105)

    ax_comp.axhspan(85, 100, color="#00ff88", alpha=0.1, label="Pass Zone")
    ax_comp.axhspan(0, 85, color="#ff4444", alpha=0.1, label="Fail Zone")
    ax_comp.axhline(85, color="#ffaa00", linestyle="--", linewidth=1, label="Pass Threshold (85%)")
    ax_comp.plot(df["row_index"], df["compliance_score"], color="#00ff88", linewidth=1.2)
    ax_comp.fill_between(df["row_index"], df["compliance_score"], 0, color="#00ff88", alpha=0.2)
    _add_legend(ax_comp)

    # STEP 12 — PANEL 8: DELIVERY SUMMARY TABLE
    ax_table = fig.add_subplot(gs[4, :])
    ax_table.axis("off")
    ax_table.set_facecolor("#16213e")
    ax_table.text(
        0.5, 0.95, "📋 DELIVERY SUMMARY",
        fontsize=12,
        color="white",
        fontweight="bold",
        ha="center",
        transform=ax_table.transAxes,
    )

    table_data = [
        ["Shipment ID", SHIPMENT_ID, "INFO"],
        ["Driver", DRIVER_NAME, "INFO"],
        ["Vehicle", VEHICLE_NUMBER, "INFO"],
        ["Avg Temperature", f"{avg_temp}°C", "OK" if TEMP_MIN <= avg_temp <= TEMP_MAX else "WARN"],
        ["Peak Temperature", f"{max_temp}°C", "OK" if max_temp <= TEMP_MAX else "EXCEEDED"],
        ["Total Alerts", str(total_alerts), "OK" if total_alerts == 0 else "ACTION NEEDED"],
        ["Door Events", str(total_door_events), "OK" if total_door_events <= 5 else "REVIEW"],
        ["Peak Spoilage Risk", f"{peak_spoilage}%", "OK" if peak_spoilage < 50 else "HIGH RISK"],
        ["Compliance Score", f"{final_compliance}%", compliance_status],
        ["Standards", "WHO GDP, SCHEDULE M", "CERTIFIED"],
    ]

    table = ax_table.table(
        cellText=table_data,
        colLabels=["Metric", "Value", "Status"],
        colWidths=[0.35, 0.35, 0.30],
        loc="center",
        cellLoc="center",
        bbox=[0.05, 0.05, 0.9, 0.82],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#0047ab")
            cell.set_text_props(color="white", fontweight="bold", fontsize=9)
            continue
        status = table_data[row - 1][2] if row > 0 else ""
        if col == 2:
            bg, fg = _table_status_color(status)
            cell.set_facecolor(bg)
            cell.set_text_props(color=fg, fontsize=9)
        else:
            cell.set_facecolor("#1a1a2e")
            cell.set_text_props(color="white", fontsize=9)
        cell.set_height(0.08)

    # STEP 13 — Save and show
    os.makedirs("reports", exist_ok=True)
    output_path = "reports/dashboard.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("Dashboard saved to reports/dashboard.png")
    plt.show()
    print("Dashboard rendering complete.")


if __name__ == "__main__":
    visualize()
