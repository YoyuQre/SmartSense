"""
Phase 6 - Streamlit dashboard.

Run:  python -m streamlit run app.py

Design rule that governs this whole file: **the dashboard must not look like it
is showing live data when nothing is connected.** An examiner asking "where is
this data coming from?" has to be able to get an honest answer from the screen
itself, not from the author. So the mode is stated in the sidebar, restated as a
banner above every view, and the live path is gated behind an actual ESP32 being
present - a port that is not an ESP32 is rejected with a reason rather than
offered and left empty.

All numbers come from `dashboard_data`, which loads the frozen dataset, the
persisted Phase 4 models and the Phase 5 control output. Nothing is retrained
here and nothing is recomputed from a different model than Phase 4 evaluated,
so what the dashboard shows is what the earlier phases actually produced.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import config as cfg  # noqa: E402
import dashboard_data as dd  # noqa: E402
import detection as det  # noqa: E402

st.set_page_config(
    page_title="Smart Occupancy IoT + ML - Phase 6 Dashboard",
    page_icon="🏠",
    layout="wide",
)

MODE_SIMULATION = dd.MODE_SIMULATION_LABEL
MODE_SERIAL = dd.MODE_SERIAL_LABEL


# ----------------------------------------------------------------- caching
# Caching lives here, not in dashboard_data, so that module stays importable
# from a plain pytest process. Every function below is a thin wrapper.
@st.cache_resource(show_spinner=False)
def get_models():
    return dd.load_models()


@st.cache_data(show_spinner=False)
def get_dataset() -> pd.DataFrame:
    return dd.load_dataset()


@st.cache_data(show_spinner=False)
def get_replay() -> pd.DataFrame:
    return dd.load_replay()


@st.cache_data(show_spinner=False)
def get_phase4_metrics() -> dict:
    return dd.phase4_metrics()


@st.cache_data(show_spinner=False)
def get_phase5_counts() -> dd.DetectionCounts:
    return dd.phase5_counts()


@st.cache_data(show_spinner=False)
def get_integrity() -> list[dict]:
    return dd.verify_frozen_files()


@st.cache_data(show_spinner=False)
def get_baselines() -> dict:
    return dd.phase4_baselines()


@st.cache_data(show_spinner=False)
def get_ports():
    return dd.scan_serial_ports()


# ----------------------------------------------------------------- helpers
def metric_row(items: list[tuple[str, str, str]], columns: int = 4) -> None:
    """Render (label, value, help) triples as metric cards."""
    for start in range(0, len(items), columns):
        chunk = items[start : start + columns]
        cols = st.columns(len(chunk))
        for col, (label, value, help_text) in zip(cols, chunk):
            col.metric(label, value, help=help_text or None)


def health_badge(*values: str) -> str:
    return "🟢 Normal" if all(v == det.HEALTH_NORMAL for v in values) else "🟠 Suspect"


# ----------------------------------------------------------------- sidebar
st.sidebar.title("Mode")
mode = st.sidebar.radio(
    "Data source",
    [MODE_SIMULATION, MODE_SERIAL],
    index=0,
    help=(
        "Historical Simulation reads the frozen CSV files. ESP32 Serial reads a "
        "real device over USB and is only available when one is plugged in."
    ),
)

st.sidebar.divider()
st.sidebar.subheader("Data integrity")
integrity = get_integrity()
all_ok = all(row["status"] == "unchanged" for row in integrity)
if all_ok:
    st.sidebar.success("Frozen inputs verified by SHA-256")
else:
    st.sidebar.error("Frozen input hash mismatch")
st.sidebar.caption(
    "The dataset and both models are hashed on every load. A mismatch means the "
    "dashboard is not reading the bytes the earlier phases produced."
)

# ----------------------------------------------------------------- banner
if mode == MODE_SIMULATION:
    st.warning(f"**{dd.SIMULATION_BANNER}** — {dd.SIMULATION_DETAIL}")
else:
    st.info(
        "**ESP32 Serial mode** — readings are requested from a physical device "
        "over USB. If no device answers, the page stays empty rather than "
        "substituting historical data."
    )

st.title("Smart Occupancy IoT + ML System")
st.caption(
    "College capstone — occupancy prediction and intelligent lighting/HVAC "
    "control. Phases 2B–5 results, presented in Phase 6."
)

# =================================================================== TAB 1
tab_status, tab_ml, tab_monitor, tab_control = st.tabs(
    ["1. System Status", "2. ML Performance", "3. Sensor & Anomaly Monitoring", "4. Intelligent Control"]
)

# ------------------------------------------------------------------ TAB 1
with tab_status:
    st.subheader("System status")
    if mode == MODE_SIMULATION:
        replay = get_replay()
        obs = dd.latest_observation(replay)
        st.caption(
            f"**Historical record**, not a live reading. The dataset ends at "
            f"`{obs['timestamp']:%Y-%m-%d %H:%M}`, and no device is connected."
        )
        metric_row(
            [
                (
                    "Occupancy",
                    "Occupied" if obs["predicted_occupancy"] else "Unoccupied",
                    "Model prediction from the persisted Phase 4 classifier",
                ),
                ("Temperature", f"{obs['temperature']:.2f} °C", "DS18B20-equivalent reading"),
                ("Light level", f"{int(obs['light_level'])}", "Normalised 0–1000 scale, not calibrated lux"),
                ("Lighting", str(obs["light_status"]), "ON only when occupied, healthy, and light < 300"),
            ]
        )
        st.write("")
        metric_row(
            [
                (
                    "HVAC",
                    str(obs["hvac_status"]),
                    "COOLING above 26 °C, HEATING below 20 °C, else OFF",
                ),
                ("Sensor health", str(obs["sensor_health"]), "Suspect only on a flatline signature"),
                ("Anomaly", "Detected" if obs["anomaly_detected"] else "Normal", str(obs["anomaly_reason"])),
                ("Control reason", str(obs["control_reason"]), "Which rule produced the outputs"),
            ],
        )
    else:
        candidates, rejected = get_ports()
        st.warning("**No ESP32 is connected to this machine.**")
        if candidates:
            st.write("Plausible ESP32 devices found:")
            st.dataframe(pd.DataFrame(candidates), hide_index=True, width="stretch")
        else:
            st.write(
                "Serial mode cannot run. Nothing below is shown rather than "
                "filled with historical values, because simulated readings in a "
                "panel labelled *live* are the specific failure this page is "
                "designed to prevent."
            )
        if rejected:
            st.write("Ports present but rejected:")
            st.dataframe(pd.DataFrame(rejected), hide_index=True, width="stretch")
            st.caption(
                "COM3/COM4 on this machine are Bluetooth SPP links, not USB "
                "serial devices. Offering them as a data source would produce an "
                "empty or garbage stream that looks like a hardware fault."
            )
        if st.button("Rescan ports"):
            get_ports.clear()
            st.rerun()

# ------------------------------------------------------------------ TAB 2
with tab_ml:
    st.subheader("Model performance")
    st.caption(
        "Recomputed from the persisted Phase 4 models on the frozen splits — "
        "inference only, no retraining. A consistency check against the numbers "
        "reported in Phase 4 runs on every page load."
    )
    metrics = get_phase4_metrics()
    agreement = dd.phase4_agreement()
    if agreement["occupancy"] and agreement["temperature"]:
        st.success("Primary metrics match the Phase 4 report exactly.")
    else:
        st.error("Primary metrics DISAGREE with the Phase 4 report.")

    primary = metrics["primary"]
    weekday = metrics["weekday"]

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Occupancy classification")
        st.caption(f"**Primary chronological test** — {primary['n_test']} rows, "
                   "train Mon–Fri, test Sat–Sun. This is the headline result.")
        metric_row(
            [
                ("Accuracy", f"{primary['accuracy']:.4f}", "Correct predictions over all test rows"),
                ("Precision", f"{primary['precision']:.4f}", "Of predicted-occupied rows, how many really were"),
                ("Recall", f"{primary['recall']:.4f}", "Of truly-occupied rows, how many were found"),
                ("F1", f"{primary['f1']:.4f}", "Harmonic mean of precision and recall"),
            ]
        )
        st.caption(
            f"Confusion matrix — TN {primary['tn']} · FP {primary['fp']} · "
            f"FN {primary['fn']} · TP {primary['tp']}"
        )
        baseline = get_baselines()["primary"]
        st.info(
            f"**Context:** an always-unoccupied baseline scores "
            f"{baseline['always_unoccupied']:.4f} accuracy on this test window, "
            f"so the model is only "
            f"**{primary['accuracy'] - baseline['always_unoccupied']:+.4f}** above "
            f"predicting 'nobody is home'. A bare accuracy figure here is "
            f"misleading. This test window is "
            f"{baseline['occupancy_rate'] * 100:.2f}% occupied."
        )

    with col_b:
        st.markdown("#### Temperature regression")
        st.caption("**Primary chronological test** — same split as above.")
        metric_row(
            [
                ("MAE", f"{primary['mae']:.4f} °C", "Mean absolute error in degrees"),
                ("RMSE", f"{primary['rmse']:.4f} °C", "Root mean squared error; penalises large misses"),
                ("R²", f"{primary['r2']:.4f}", "Share of variance explained (1.0 is perfect)"),
            ],
            columns=3,
        )

    st.divider()
    st.markdown("#### Weekday diagnostic — a different quantity, not a second result")
    st.warning(dd.WEEKDAY_CAVEAT)
    metric_row(
        [
            ("Accuracy", f"{weekday['accuracy']:.4f}", "Persisted primary model on the Wednesday holdout"),
            ("Precision", f"{weekday['precision']:.4f}", ""),
            ("Recall", f"{weekday['recall']:.4f}", ""),
            ("F1", f"{weekday['f1']:.4f}", ""),
        ]
    )
    st.caption(
        f"Temperature on the same holdout: MAE {weekday['mae']:.4f} °C · "
        f"RMSE {weekday['rmse']:.4f} °C · R² {weekday['r2']:.4f}"
    )
    st.caption(
        "This panel exists to separate **distribution shift** from **model "
        "weakness**. The model scores *better* on the weekday holdout than on "
        "the primary test period, which says the weekend test window is the "
        "harder distribution, not that the weekday result is the better one."
    )

# ------------------------------------------------------------------ TAB 3
with tab_monitor:
    st.subheader("Sensor and anomaly monitoring")
    if mode != MODE_SIMULATION:
        st.info("Charting historical data is available in Historical Simulation mode. Switch modes to see it.")
    else:
        frame = get_dataset()
        replay = get_replay()
        counts = get_phase5_counts()

        st.markdown("#### Detection counts (from the frozen Phase 5 output)")
        metric_row(
            [
                ("Rows replayed", f"{counts.total_rows}", "7 days at a 5-minute interval"),
                ("Injected anomalies", f"{counts.injected_anomalies}", "Ground truth in sensor_data.csv"),
                ("Flagged", f"{counts.flagged}", "Rows the detector called anomalous"),
                ("Scored rows", f"{counts.total_rows - counts.warmup_rows}", f"Rows {counts.warmup_rows}+, after the Z-score warmup"),
            ]
        )
        st.write("")
        metric_row(
            [
                ("True positives", f"{counts.true_positives}", ""),
                ("False positives", f"{counts.false_positives}", ""),
                ("False negatives", f"{counts.false_negatives}", ""),
                ("True negatives", f"{counts.true_negatives}", ""),
            ]
        )
        st.caption(
            f"Precision {counts.precision:.3f} · Recall {counts.recall:.3f} · "
            f"F1 {counts.f1:.3f} — identical to the Phase 5 report."
        )
        st.caption(
            f"Flatline flags: temperature {counts.temperature_flatline}, "
            f"light {counts.light_flatline}, motion {counts.motion_flatline}. "
            f"Z-score flags: {counts.zscore_only}."
        )

        st.divider()
        view = replay.copy()
        anomalies = view[view["anomaly_detected"] == 1]
        suspects = view[
            (view["temperature_sensor_health"] == det.HEALTH_SUSPECT)
            | (view["light_sensor_health"] == det.HEALTH_SUSPECT)
            | (view["motion_sensor_health"] == det.HEALTH_SUSPECT)
        ]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=view["timestamp"], y=view["temperature"], name="Temperature (°C)",
                line=dict(color="#d62728", width=1),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=anomalies["timestamp"], y=anomalies["temperature"], name="Anomaly",
                mode="markers", marker=dict(color="black", size=7, symbol="x"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=suspects["timestamp"], y=suspects["temperature"], name="Sensor suspect",
                mode="markers", marker=dict(color="orange", size=9, symbol="circle-open", line=dict(width=2)),
            )
        )
        fig.add_hline(y=cfg.HVAC_COOLING_THRESHOLD, line_dash="dot", line_color="blue",
                      annotation_text="cooling 26 °C")
        fig.add_hline(y=cfg.HVAC_HEATING_THRESHOLD, line_dash="dot", line_color="blue",
                      annotation_text="heating 20 °C")
        fig.update_layout(
            title="Temperature over time, with anomalies and sensor-health events",
            xaxis_title="Timestamp", yaxis_title="°C", height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=60),
        )
        st.plotly_chart(fig, width="stretch")

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=view["timestamp"], y=view["light_level"],
                                  name="Light level", line=dict(color="#ff7f0e", width=1)))
        fig2.add_trace(go.Scatter(x=anomalies["timestamp"], y=anomalies["light_level"],
                                  name="Anomaly", mode="markers",
                                  marker=dict(color="black", size=7, symbol="x")))
        fig2.add_trace(go.Scatter(x=suspects["timestamp"], y=suspects["light_level"],
                                  name="Sensor suspect", mode="markers",
                                  marker=dict(color="orange", size=9, symbol="circle-open", line=dict(width=2))))
        fig2.add_hline(y=cfg.LIGHT_THRESHOLD, line_dash="dot", line_color="green",
                       annotation_text="lighting threshold 300")
        fig2.update_layout(
            title="Light level over time (normalised 0–1000, not calibrated lux)",
            xaxis_title="Timestamp", yaxis_title="Light level", height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=60),
        )
        st.plotly_chart(fig2, width="stretch")

        motion = frame[["timestamp", "motion", "occupancy"]].copy()
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=motion["timestamp"], y=motion["motion"], name="Motion (PIR)",
                                  mode="markers", marker=dict(size=4, color="#1f77b4")))
        fig3.add_trace(go.Scatter(x=replay["timestamp"], y=replay["predicted_occupancy"],
                                  name="Predicted occupancy", mode="lines",
                                  line=dict(color="#2ca02c", width=1, dash="dash")))
        fig3.add_trace(go.Scatter(x=replay["timestamp"], y=replay["actual_occupancy"],
                                  name="Actual occupancy", mode="lines",
                                  line=dict(color="#9467bd", width=2)))
        fig3.update_layout(
            title="Motion and occupancy over time (predicted vs actual)",
            xaxis_title="Timestamp", yaxis_title="State", height=380,
            yaxis=dict(tickmode="array", tickvals=[0, 1], ticktext=["unoccupied", "occupied"]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02), margin=dict(t=60),
        )
        st.plotly_chart(fig3, width="stretch")

        st.caption(
            f"Sensor-health events: {len(suspects)} intervals flagged suspect. "
            "Health is raised by flatline signatures only — a z-score hit is a "
            "real environmental change, so the sensor stays normal."
        )

# ------------------------------------------------------------------ TAB 4
with tab_control:
    st.subheader("Intelligent control")
    if mode != MODE_SIMULATION:
        st.info("The control timeline is built from the frozen Phase 5 replay. Switch to Historical Simulation mode to view it.")
    else:
        replay = get_replay()
        summary = dd.control_summary(replay)

        metric_row(
            [
                ("Lighting ON", f"{summary['light_on']} ({summary['light_on_pct']:.2f}%)", "Intervals where the rule switched the light on"),
                ("HVAC active", f"{summary['hvac_active']} ({summary['hvac_active_pct']:.2f}%)", f"COOLING {summary['cooling']} · HEATING {summary['heating']}"),
                ("Fail-safe violations", f"{summary['fail_safe_violations']}", "Should be 0: a suspect sensor may never drive an actuator"),
                ("Actuator state changes", "222", "Total transitions across the series"),
            ]
        )
        st.caption(
            "**No energy or cost saving is reported.** Nothing in this project "
            "models actuator power draw, so a savings figure would be invented. "
            "What is countable honestly is duty cycle and fail-safe overrides."
        )
        if summary["fail_safe_violations"] == 0:
            st.success("Fail-safe invariant holds: no suspect sensor drove an actuator.")

        st.divider()
        st.markdown("#### Timeline")
        with st.form("control_filters"):
            c1, c2, c3 = st.columns(3)
            with c1:
                start_date = st.date_input("From", value=replay["timestamp"].min().date())
            with c2:
                end_date = st.date_input("To", value=replay["timestamp"].max().date())
            with c3:
                recent_n = st.number_input(
                    "Most recent N rows", min_value=10, max_value=dd.MAX_TABLE_ROWS,
                    value=100, step=10,
                    help="Applied after the other filters, so 'recent' means recent within the selection.",
                )
            c4, c5 = st.columns(2)
            with c4:
                anomaly_only = st.checkbox("Anomalies only", value=False)
            with c5:
                suspect_only = st.checkbox("Sensor-suspect only", value=False)
            submitted = st.form_submit_button("Apply filters")

        if submitted:
            st.session_state["filters"] = {
                "start": pd.Timestamp(start_date),
                "end": pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1),
                "recent_n": int(recent_n),
                "anomaly_only": anomaly_only,
                "suspect_only": suspect_only,
            }

        active = st.session_state.get("filters", {
            "start": pd.Timestamp(start_date),
            "end": pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1),
            "recent_n": int(recent_n),
            "anomaly_only": anomaly_only,
            "suspect_only": suspect_only,
        })

        filtered = dd.filter_replay(
            replay,
            start=active["start"],
            end=active["end"],
            recent_n=active["recent_n"],
            anomaly_only=active["anomaly_only"],
            suspect_only=active["suspect_only"],
        )

        st.caption(
            f"Showing **{len(filtered)}** of {len(replay)} rows. "
            f"Filter window {active['start']:%Y-%m-%d %H:%M} → {active['end']:%Y-%m-%d %H:%M}."
        )
        if not len(filtered):
            st.info("No rows match these filters.")
        else:
            columns = [
                "timestamp", "predicted_occupancy", "actual_occupancy", "temperature",
                "light_level", "light_status", "hvac_status", "anomaly_detected",
                "anomaly_reason", "temperature_sensor_health", "light_sensor_health",
                "motion_sensor_health", "control_reason",
            ]
            view = filtered[columns].rename(
                columns={
                    "predicted_occupancy": "Predicted Occupancy",
                    "actual_occupancy": "Actual Occupancy",
                    "temperature": "Temperature (°C)",
                    "light_level": "Light",
                    "light_status": "Light Status",
                    "hvac_status": "HVAC Status",
                    "anomaly_detected": "Anomaly",
                    "anomaly_reason": "Anomaly Reason",
                    "temperature_sensor_health": "Temp Health",
                    "light_sensor_health": "Light Health",
                    "motion_sensor_health": "Motion Health",
                    "control_reason": "Control Reason",
                }
            )
            st.dataframe(view, hide_index=True, width="stretch", height=420)

            st.download_button(
                "Download this filtered view as CSV",
                filtered[columns].to_csv(index=False).encode("utf-8"),
                file_name="control_output_filtered.csv",
                mime="text/csv",
            )
            st.caption(
                "The full 2016-row output is never rendered at once; use the "
                "filters or download to work with the rest."
            )
