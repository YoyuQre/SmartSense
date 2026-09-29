"""
Phase 6 - dashboard tests.

The Phase 6 acceptance criteria are mostly statements about truthfulness, so
most of these tests check that the dashboard reports what the earlier phases
actually produced rather than something merely plausible. Three of them are
source-level checks (no retraining, no fabricated live data, clear mode label)
because those properties cannot be observed from rendered output.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import control_logic as ctl  # noqa: E402
import dashboard_data as dd  # noqa: E402
import detection as det  # noqa: E402

APP_PATH = ROOT / "app.py"


@pytest.fixture(scope="module")
def replay() -> pd.DataFrame:
    return dd.load_replay()


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    return dd.load_dataset()


# ==========================================================================
# Acceptance: no ML retraining inside the dashboard
# ==========================================================================
FORBIDDEN_CALLS = {"fit", "fit_transform", "partial_fit", "train_test_split", "GridSearchCV", "cross_val_score"}


def _called_attributes(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                called.add(func.attr)
            elif isinstance(func, ast.Name):
                called.add(func.id)
    return called


def test_app_never_trains_a_model():
    """A dashboard that refits on load is not a dashboard, it is a training run."""
    offenders = _called_attributes(APP_PATH) & FORBIDDEN_CALLS
    assert not offenders, f"app.py calls training APIs: {sorted(offenders)}"


def test_dashboard_data_never_trains_a_model():
    offenders = _called_attributes(ROOT / "src" / "dashboard_data.py") & FORBIDDEN_CALLS
    assert not offenders, f"dashboard_data.py calls training APIs: {sorted(offenders)}"


def test_dashboard_data_does_not_import_model_training_helpers():
    """`modeling` is imported for its metric functions, not its fitting ones."""
    source = (ROOT / "src" / "dashboard_data.py").read_text(encoding="utf-8")
    for forbidden in ["fit_occupancy", "fit_temperature", "tune_decision_tree"]:
        assert forbidden not in source, (
            f"dashboard_data.py references {forbidden}; metrics must come from "
            "the persisted models, not a fresh fit"
        )


def test_models_are_loaded_from_disk():
    source = (ROOT / "src" / "dashboard_data.py").read_text(encoding="utf-8")
    assert "joblib.load" in source
    assert re.search(r"joblib\.load\(\s*OCCUPANCY_MODEL", source)
    assert re.search(r"joblib\.load\(\s*TEMPERATURE_MODEL", source)


# ==========================================================================
# Acceptance: metrics match Phase 4
# ==========================================================================
def test_primary_metrics_match_the_phase4_report():
    metrics = dd.phase4_metrics()["primary"]
    expected = dd.PHASE4_PRIMARY
    for key in ["accuracy", "precision", "recall", "f1"]:
        assert metrics[key] == pytest.approx(expected[key], abs=5e-4), key
    for key in ["mae", "rmse", "r2"]:
        assert metrics[key] == pytest.approx(expected[key], abs=5e-4), key
    for key in ["tn", "fp", "fn", "tp", "n_test"]:
        assert metrics[key] == expected[key], key


def test_phase4_agreement_check_passes():
    agreement = dd.phase4_agreement()
    assert agreement["occupancy"] and agreement["temperature"]


def test_baseline_is_computed_from_the_test_split_not_hardcoded():
    """Regression guard: the training occupancy rate is not the test rate."""
    baselines = dd.phase4_baselines()
    assert baselines["primary"]["always_unoccupied"] == pytest.approx(0.7361, abs=5e-4)
    metrics = dd.phase4_metrics()["primary"]
    gain = metrics["accuracy"] - baselines["primary"]["always_unoccupied"]
    assert gain == pytest.approx(0.0417, abs=5e-4), (
        "the model is only ~+0.042 above a trivial baseline; a large apparent "
        "gain means the wrong baseline rate was used"
    )


def test_weekday_panel_is_labelled_as_a_different_quantity():
    """The weekday number is a different model/split pairing, so must be labelled."""
    assert "not the same quantity" in dd.WEEKDAY_CAVEAT
    assert "never persisted" in dd.WEEKDAY_CAVEAT
    assert dd.PHASE4_WEEKDAY_SEPARATE_MODEL["f1"] != pytest.approx(
        dd.phase4_metrics()["weekday"]["f1"], abs=5e-4
    )


# ==========================================================================
# Acceptance: anomaly counts match Phase 5
# ==========================================================================
def test_anomaly_counts_match_phase5():
    counts = dd.phase5_counts()
    assert counts.total_rows == 2016
    assert counts.injected_anomalies == 40
    # Phase 5 reported, on rows 288+:
    assert counts.true_positives == 25
    assert counts.false_positives == 10
    assert counts.false_negatives == 8
    assert counts.precision == pytest.approx(0.714, abs=5e-3)
    assert counts.recall == pytest.approx(0.758, abs=5e-3)
    assert counts.f1 == pytest.approx(0.735, abs=5e-3)


def test_flatline_counts_match_phase5():
    counts = dd.phase5_counts()
    assert counts.temperature_flatline == 12
    assert counts.light_flatline == 8
    assert counts.motion_flatline == 0
    assert counts.zscore_only == 21


def test_counts_are_read_from_the_frozen_output_not_recomputed():
    """Dashboard must display Phase 5's answer, not a fresh derivation."""
    source = (ROOT / "src" / "dashboard_data.py").read_text(encoding="utf-8")
    assert "REPLAY_PATH" in source
    # detect() is not called in dashboard_data; counts come from the CSV.
    assert "det.detect(" not in source


# ==========================================================================
# Acceptance: historical controls match control_output.csv
# ==========================================================================
def test_replay_rows_are_genuine_phase5_control_decisions(replay):
    """Re-derive the control decision from each row's own inputs.

    If the CSV's statuses did not follow the tested control rules, the
    dashboard would be presenting labels with no logic behind them.
    """
    sample = replay.iloc[::37]  # spread across the whole series
    mismatches = 0
    for _, row in sample.iterrows():
        decision = ctl.decide_control(
            int(row["predicted_occupancy"]),
            float(row["temperature"]),
            float(row["light_level"]),
            row["temperature_sensor_health"],
            row["light_sensor_health"],
        )
        if decision.light_status != row["light_status"] or decision.hvac_status != row["hvac_status"]:
            mismatches += 1
    assert mismatches == 0, f"{mismatches} replay rows disagree with the control rules"


def test_control_summary_matches_the_file(replay):
    summary = dd.control_summary(replay)
    assert summary["rows"] == len(replay)
    assert summary["light_on"] == int((replay["light_status"] == "ON").sum())
    assert summary["cooling"] == int((replay["hvac_status"] == "COOLING").sum())
    assert summary["heating"] == int((replay["hvac_status"] == "HEATING").sum())
    assert summary["fail_safe_violations"] == 0


def test_latest_observation_is_a_historical_record(replay):
    obs = dd.latest_observation(replay)
    assert obs["timestamp"] == replay["timestamp"].iloc[-1]
    assert obs["timestamp"] < pd.Timestamp("2026-01-12")  # the dataset ends in Jan
    assert obs["sensor_health"] in {"Normal", "Suspect"}


# ==========================================================================
# Acceptance: no fabricated live data
# ==========================================================================
def test_serial_mode_rejects_non_esp32_ports():
    candidates, rejected = dd.scan_serial_ports()
    for entry in rejected:
        assert "not an ESP32" in entry["reason"]
    # This machine only has Bluetooth SPP ports, so nothing may be offered.
    assert candidates == [], (
        "an ESP32 candidate appeared without hardware; the port matcher is "
        "too permissive"
    )


def test_port_matcher_rejects_bluetooth_and_accepts_esp32_descriptions():
    """Unit-test the matcher without needing real hardware."""
    from serial.tools import list_ports

    class FakePort:
        def __init__(self, device, description, vid, pid):
            self.device, self.description, self.vid, self.pid = device, description, vid, pid

    original = list_ports.comports
    try:
        list_ports.comports = lambda: [
            FakePort("COM3", "Standard Serial over Bluetooth link (COM3)", None, None),
            FakePort("COM7", "Silicon Labs CP210x USB to UART Bridge (COM7)", 0x10C4, 0xEA60),
            FakePort("COM9", "Espressif USB JTAG/serial debug unit (COM9)", 0x303A, 0x1001),
        ]
        candidates, rejected = dd.scan_serial_ports()
        devices = {c["device"] for c in candidates}
        assert devices == {"COM7", "COM9"}, devices
        assert [r["device"] for r in rejected] == ["COM3"]
    finally:
        list_ports.comports = original


def test_app_disables_live_views_when_no_device_is_present():
    source = APP_PATH.read_text(encoding="utf-8")
    assert "No ESP32 is connected" in source
    assert "stays empty rather than" in source
    # The historical view must be gated on the simulation mode, not always shown.
    assert 'if mode != MODE_SIMULATION' in source


def test_serial_protocol_parser_handles_the_real_record_shape():
    text = "\n".join([
        "======================================",
        " Smart Occupancy IoT System",
        "[INIT] DS18B20 found on GPIO 4, devices=1",
        "DATA,5000,0,742,24.31,0,0",
        "DATA,10000,1,218,24.38,1,0",
        "ERR,TEMP_INVALID,raw=-127.00,action=HVAC_FORCED_OFF",
        "DATA,15000,1,185,24.44,1,0",
    ])
    records, errors = dd.parse_serial_lines(text)
    assert len(records) == 3
    assert records[0] == {
        "timestamp_ms": 5000.0, "motion": 0.0, "light_level": 742.0,
        "temperature": 24.31, "light_status": 0.0, "hvac_status": 0.0,
    }
    assert len(errors) == 1 and "TEMP_INVALID" in errors[0]


def test_serial_parser_reports_garbage_instead_of_hiding_it():
    records, errors = dd.parse_serial_lines("DATA,1000,0\nNOT_A_RECORD\nDATA,2000,0,5,20.0,0,0")
    assert len(records) == 1
    assert len(errors) == 2, "malformed lines must be surfaced, not silently dropped"


# ==========================================================================
# Acceptance: dashboard reads frozen datasets, source hash unchanged
# ==========================================================================
def test_frozen_files_are_unchanged():
    rows = dd.verify_frozen_files()
    assert rows, "no frozen files were checked"
    for row in rows:
        assert row["status"] == "unchanged", f"{row['file']} is {row['status']}"


def test_dataset_hash_still_matches_phase2():
    import hashlib

    digest = hashlib.sha256((ROOT / "data" / "sensor_data.csv").read_bytes()).hexdigest().upper()
    assert digest == dd.FROZEN_SHA256["data/sensor_data.csv"]


def test_dashboard_reads_the_frozen_paths():
    source = (ROOT / "src" / "dashboard_data.py").read_text(encoding="utf-8")
    assert 'ROOT / "data" / "sensor_data.csv"' in source
    assert 'ROOT / "data" / "control_output.csv"' in source


# ==========================================================================
# Acceptance: filters work
# ==========================================================================
def test_recent_n_limits_rows(replay):
    for n in [10, 50, 100]:
        assert len(dd.filter_replay(replay, recent_n=n)) == n


def test_recent_n_keeps_the_latest_rows(replay):
    filtered = dd.filter_replay(replay, recent_n=5)
    assert list(filtered["timestamp"]) == list(replay["timestamp"].iloc[-5:])


def test_recent_n_never_exceeds_available_rows(replay):
    assert len(dd.filter_replay(replay, recent_n=5000)) == len(replay)


def test_date_range_filter(replay):
    start = replay["timestamp"].iloc[0]
    end = start + pd.Timedelta(hours=5)
    filtered = dd.filter_replay(replay, start=start, end=end, recent_n=None)
    assert len(filtered) == 61  # inclusive of both endpoints, 5-minute interval
    assert filtered["timestamp"].min() == start
    assert filtered["timestamp"].max() == end


def test_anomaly_only_filter(replay):
    filtered = dd.filter_replay(replay, anomaly_only=True, recent_n=None)
    assert (filtered["anomaly_detected"] == 1).all()
    assert len(filtered) == int((replay["anomaly_detected"] == 1).sum())


def test_suspect_only_filter(replay):
    filtered = dd.filter_replay(replay, suspect_only=True, recent_n=None)
    assert len(filtered) > 0
    suspect = (
        (filtered["temperature_sensor_health"] == det.HEALTH_SUSPECT)
        | (filtered["light_sensor_health"] == det.HEALTH_SUSPECT)
        | (filtered["motion_sensor_health"] == det.HEALTH_SUSPECT)
    )
    assert suspect.all()


def test_filters_combine(replay):
    filtered = dd.filter_replay(replay, anomaly_only=True, recent_n=3)
    assert len(filtered) == 3
    assert (filtered["anomaly_detected"] == 1).all()


def test_recent_n_applies_after_other_filters(replay):
    """"Recent" must mean recent within the selection, not within the file."""
    everything = dd.filter_replay(replay, recent_n=None)
    top_three = dd.filter_replay(replay, recent_n=3)
    assert list(top_three["timestamp"]) == list(everything["timestamp"].iloc[-3:])


def test_table_is_capped_so_the_page_is_not_dumped(replay):
    assert dd.MAX_TABLE_ROWS < len(replay)
    assert len(dd.filter_replay(replay, recent_n=dd.MAX_TABLE_ROWS)) == dd.MAX_TABLE_ROWS


# ==========================================================================
# Acceptance: clear Simulation Mode label
# ==========================================================================
def test_simulation_banner_is_explicit():
    assert "Simulation Mode" in dd.SIMULATION_BANNER
    assert "historical" in dd.SIMULATION_BANNER.lower()
    assert "No ESP32 is streaming" in dd.SIMULATION_DETAIL


def test_app_shows_the_simulation_banner_and_defaults_to_it():
    source = APP_PATH.read_text(encoding="utf-8")
    assert "SIMULATION_BANNER" in source
    assert "SIMULATION_DETAIL" in source
    # Mode labels are delegated to dashboard_data so the tests assert on the
    # same strings the UI renders.
    assert "MODE_SIMULATION = dd.MODE_SIMULATION_LABEL" in source
    assert "MODE_SERIAL = dd.MODE_SERIAL_LABEL" in source
    # Historical Simulation must be first in the radio, i.e. the default.
    assert "[MODE_SIMULATION, MODE_SERIAL]" in source
    assert "index=0" in source


def test_app_does_not_claim_energy_savings():
    source = APP_PATH.read_text(encoding="utf-8")
    assert "No energy or cost saving is reported" in source


# ==========================================================================
# The app itself runs
# ==========================================================================
def test_app_runs_without_exceptions_in_simulation_mode():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(APP_PATH), default_timeout=180)
    app.run()
    assert not app.exception, [str(e.value) for e in app.exception]
    assert len(app.tabs) == 4
    assert any("Simulation Mode" in w.value for w in app.warning)
    assert any("match the Phase 4 report" in s.value for s in app.success)
    assert any("Frozen inputs verified" in s.value for s in app.success)


def test_app_runs_and_refuses_to_fake_data_in_serial_mode():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(APP_PATH), default_timeout=180)
    app.run()
    app.sidebar.radio[0].set_value(dd.MODE_SERIAL_LABEL).run()
    assert not app.exception, [str(e.value) for e in app.exception]
    assert any("No ESP32 is connected" in w.value for w in app.warning)
    # No status cards may be rendered, because there is no data to fill them.
    assert not any("Occupied" in m.value for m in app.metric)
