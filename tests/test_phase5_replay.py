"""
Phase 5 - integration tests over the generated replay output.

The unit tests in test_control_logic.py prove the rules one case at a time.
These tests instead check the properties that must hold across the whole
2016-row replay, which a per-case test cannot see. The fail-safe invariant is
the important one: it is the property that makes the system safe to leave
running, so it is asserted over every row rather than sampled.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import control_logic as ctl  # noqa: E402
import detection as det  # noqa: E402
import preprocessing as pp  # noqa: E402

OUTPUT_PATH = ROOT / "data" / "control_output.csv"
SOURCE_PATH = ROOT / "data" / "sensor_data.csv"

REQUIRED_COLUMNS = [
    "timestamp",
    "predicted_occupancy",
    "actual_occupancy",
    "temperature",
    "light_level",
    "light_status",
    "hvac_status",
    "anomaly_detected",
    "anomaly_reason",
    "temperature_sensor_health",
    "light_sensor_health",
    "motion_sensor_health",
    "control_reason",
]


@pytest.fixture(scope="module")
def replay() -> pd.DataFrame:
    if not OUTPUT_PATH.exists():
        pytest.skip(
            "data/control_output.csv not present; run `python src/phase5_run.py` first"
        )
    frame = pd.read_csv(OUTPUT_PATH)
    # The CSV round-trip returns strings; parse so comparisons against the
    # source frame compare timestamps rather than their formatting.
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------
def test_replay_has_the_required_columns(replay):
    missing = [c for c in REQUIRED_COLUMNS if c not in replay.columns]
    assert not missing, f"missing required columns: {missing}"


def test_replay_covers_every_source_row(replay):
    """One output row per input row, in order, with no rows invented or dropped."""
    source = pp.load_dataset()
    assert len(replay) == len(source) == 2016
    assert list(replay["timestamp"]) == list(source["timestamp"])
    assert replay["temperature"].to_numpy() == pytest.approx(
        source["temperature"].to_numpy()
    )
    assert replay["light_level"].to_numpy() == pytest.approx(
        source["light_level"].to_numpy()
    )
    assert np.array_equal(
        replay["actual_occupancy"].to_numpy(), source["occupancy"].to_numpy()
    )


def test_replay_contains_no_missing_values(replay):
    assert not replay.isna().any().any(), "a control decision must never be blank"


def test_status_and_health_vocabularies_are_closed(replay):
    assert set(replay["light_status"]) <= {ctl.LIGHT_ON, ctl.LIGHT_OFF}
    assert set(replay["hvac_status"]) <= {
        ctl.HVAC_COOLING,
        ctl.HVAC_HEATING,
        ctl.HVAC_OFF,
    }
    for column in [
        "temperature_sensor_health",
        "light_sensor_health",
        "motion_sensor_health",
    ]:
        assert set(replay[column]) <= {det.HEALTH_NORMAL, det.HEALTH_SUSPECT}
    assert set(replay["anomaly_reason"]) <= {
        det.REASON_NONE,
        det.REASON_ZSCORE,
        det.REASON_TEMP_FLAT,
        det.REASON_LIGHT_FLAT,
        det.REASON_MOTION_FLAT,
    }
    assert set(replay["predicted_occupancy"]) <= {0, 1}
    assert set(replay["actual_occupancy"]) <= {0, 1}
    assert set(replay["anomaly_detected"]) <= {0, 1}


# --------------------------------------------------------------------------
# Safety invariants, over every row
# --------------------------------------------------------------------------
def test_fail_safe_never_acts_on_a_suspect_sensor(replay):
    """The core safety property, asserted on all 2016 rows.

    If the light sensor is suspect the light must be OFF, and if the
    temperature sensor is suspect the HVAC must be OFF. A single violation
    would mean the fail-safe can be talked out of.
    """
    light_suspect = replay["light_sensor_health"] == det.HEALTH_SUSPECT
    if light_suspect.any():
        assert (replay.loc[light_suspect, "light_status"] == ctl.LIGHT_OFF).all(), (
            f"{(replay.loc[light_suspect, 'light_status'] != ctl.LIGHT_OFF).sum()} "
            "row(s) drove the light from a suspect sensor"
        )

    temp_suspect = replay["temperature_sensor_health"] == det.HEALTH_SUSPECT
    if temp_suspect.any():
        assert (replay.loc[temp_suspect, "hvac_status"] == ctl.HVAC_OFF).all(), (
            f"{(replay.loc[temp_suspect, 'hvac_status'] != ctl.HVAC_OFF).sum()} "
            "row(s) drove the HVAC from a suspect sensor"
        )


def test_suspect_sensors_actually_occur(replay):
    """Guard against the previous test passing vacuously.

    If no sensor is ever suspect, the fail-safe assertion above proves nothing.
    This pins the fact that the replay does exercise the fail-safe path.
    """
    assert (replay["temperature_sensor_health"] == det.HEALTH_SUSPECT).any()
    assert (replay["light_sensor_health"] == det.HEALTH_SUSPECT).any()


def test_light_never_on_while_unoccupied(replay):
    on = replay["light_status"] == ctl.LIGHT_ON
    if on.any():
        assert (replay.loc[on, "predicted_occupancy"] == 1).all()
        assert (replay.loc[on, "light_level"] < 300).all()


def test_hvac_never_runs_while_unoccupied(replay):
    active = replay["hvac_status"].isin([ctl.HVAC_COOLING, ctl.HVAC_HEATING])
    if active.any():
        assert (replay.loc[active, "predicted_occupancy"] == 1).all()


def test_heating_and_cooling_never_co_occur(replay):
    """`hvac_status` is single-valued, so both-on is not representable."""
    assert not (
        (replay["hvac_status"] == ctl.HVAC_COOLING)
        & (replay["hvac_status"] == ctl.HVAC_HEATING)
    ).any()


def test_every_row_has_a_control_reason(replay):
    assert (replay["control_reason"].astype(str).str.len() > 0).all()


# --------------------------------------------------------------------------
# The source dataset was not disturbed
# --------------------------------------------------------------------------
def test_source_dataset_is_unchanged():
    """Phase 5 must not have written to the frozen Phase 2/3 dataset."""
    import hashlib

    expected = (
        "2D38D7416FDD8D80FE78C52E383BCDC50943722CEB346AD9B2B46ED5BED7D091"
    )
    actual = hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest().upper()
    assert actual == expected, "data/sensor_data.csv was modified in Phase 5"


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------
def test_detection_is_deterministic():
    frame = pp.load_dataset()
    first = det.detect(frame)
    second = det.detect(frame)
    assert first.anomaly_detected.tolist() == second.anomaly_detected.tolist()
    assert first.anomaly_reason == second.anomaly_reason


def test_replay_script_is_reproducible():
    """Re-running the replay must not change the numbers it reports.

    This catches a regression where a refit or an unseeded step leaked into
    the pipeline, which would otherwise be invisible until the next report.
    """
    before = OUTPUT_PATH.read_bytes()
    completed = subprocess.run(
        [sys.executable, str(ROOT / "src" / "phase5_run.py")],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    after = OUTPUT_PATH.read_bytes()
    assert before == after, "re-running phase5_run.py changed the output file"
