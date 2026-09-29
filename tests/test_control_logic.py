"""
Phase 5 - control truth-table tests.

Every case in the Phase 5 specification has an explicit test here. The control
rules are pure functions, so these are exact assertions with no tolerance and
no randomness: a failure means the rule is wrong, not that the data moved.

Run:  python -m pytest tests -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import config as cfg  # noqa: E402
from control_logic import (  # noqa: E402
    REASON_COOLING,
    REASON_HEATING,
    REASON_LIGHT_SUSPECT,
    REASON_OCCUPIED_BRIGHT,
    REASON_OCCUPIED_DARK,
    REASON_TEMP_SUSPECT,
    REASON_UNOCCUPIED,
    REASON_UNOCCUPIED_HVAC,
    REASON_WITHIN_BAND,
    decide_control,
    decide_hvac,
    decide_lighting,
)
from detection import HEALTH_NORMAL, HEALTH_SUSPECT  # noqa: E402
from detection import HVAC_COOLING, HVAC_HEATING, HVAC_OFF, LIGHT_OFF, LIGHT_ON  # noqa: E402

DARK = cfg.LIGHT_THRESHOLD - 100       # comfortably below threshold
BRIGHT = cfg.LIGHT_THRESHOLD + 100     # comfortably above threshold
AT_THRESHOLD = cfg.LIGHT_THRESHOLD
HOT = cfg.HVAC_COOLING_THRESHOLD + 1.0
COLD = cfg.HVAC_HEATING_THRESHOLD - 1.0


# --------------------------------------------------------------------------
# Lighting truth table
# --------------------------------------------------------------------------
def test_occupied_and_dark_light_on():
    status, reason = decide_lighting(1, DARK, HEALTH_NORMAL)
    assert status == LIGHT_ON
    assert reason == REASON_OCCUPIED_DARK


def test_occupied_and_bright_light_off():
    status, reason = decide_lighting(1, BRIGHT, HEALTH_NORMAL)
    assert status == LIGHT_OFF
    assert reason == REASON_OCCUPIED_BRIGHT


def test_unoccupied_and_dark_light_off():
    status, reason = decide_lighting(0, DARK, HEALTH_NORMAL)
    assert status == LIGHT_OFF
    assert reason == REASON_UNOCCUPIED


def test_unoccupied_and_bright_light_off():
    status, reason = decide_lighting(0, BRIGHT, HEALTH_NORMAL)
    assert status == LIGHT_OFF
    assert reason == REASON_UNOCCUPIED


def test_light_exactly_at_threshold_is_off():
    """Strict `<`: a reading exactly at the threshold counts as sufficient light."""
    assert AT_THRESHOLD == cfg.LIGHT_THRESHOLD
    status, reason = decide_lighting(1, AT_THRESHOLD, HEALTH_NORMAL)
    assert status == LIGHT_OFF
    assert reason == REASON_OCCUPIED_BRIGHT


def test_light_one_below_threshold_is_on():
    status, _ = decide_lighting(1, cfg.LIGHT_THRESHOLD - 1, HEALTH_NORMAL)
    assert status == LIGHT_ON


def test_suspect_light_sensor_fails_safe_to_off():
    """Fails safe even when the measurement alone would switch the light on."""
    status, reason = decide_lighting(1, DARK, HEALTH_SUSPECT)
    assert status == LIGHT_OFF
    assert reason == REASON_LIGHT_SUSPECT


# --------------------------------------------------------------------------
# Cooling / HVAC truth table
# --------------------------------------------------------------------------
def test_occupied_and_hot_cooling_on():
    status, reason = decide_hvac(1, HOT, HEALTH_NORMAL)
    assert status == HVAC_COOLING
    assert reason == REASON_COOLING


def test_occupied_temperature_exactly_at_cooling_threshold_is_off():
    """Strict `>`: exactly at the threshold does not start cooling."""
    assert cfg.HVAC_COOLING_THRESHOLD == 26.0
    status, reason = decide_hvac(1, cfg.HVAC_COOLING_THRESHOLD, HEALTH_NORMAL)
    assert status == HVAC_OFF
    assert reason == REASON_WITHIN_BAND


def test_occupied_below_threshold_hvac_off():
    status, reason = decide_hvac(1, cfg.HVAC_COOLING_THRESHOLD - 5, HEALTH_NORMAL)
    assert status == HVAC_OFF
    assert reason == REASON_WITHIN_BAND


def test_unoccupied_and_hot_hvac_off():
    status, reason = decide_hvac(0, HOT, HEALTH_NORMAL)
    assert status == HVAC_OFF
    assert reason == REASON_UNOCCUPIED_HVAC


def test_suspect_temperature_sensor_fails_safe_to_off():
    status, reason = decide_hvac(1, HOT, HEALTH_SUSPECT)
    assert status == HVAC_OFF
    assert reason == REASON_TEMP_SUSPECT


# --------------------------------------------------------------------------
# Heating truth table
# --------------------------------------------------------------------------
def test_occupied_and_cold_heating_on():
    status, reason = decide_hvac(1, COLD, HEALTH_NORMAL)
    assert status == HVAC_HEATING
    assert reason == REASON_HEATING


def test_occupied_temperature_exactly_at_heating_threshold_is_off():
    status, reason = decide_hvac(1, cfg.HVAC_HEATING_THRESHOLD, HEALTH_NORMAL)
    assert status == HVAC_OFF
    assert reason == REASON_WITHIN_BAND


def test_unoccupied_and_cold_hvac_off():
    status, reason = decide_hvac(0, COLD, HEALTH_NORMAL)
    assert status == HVAC_OFF
    assert reason == REASON_UNOCCUPIED_HVAC


def test_suspect_temperature_sensor_fails_safe_even_when_cold():
    status, reason = decide_hvac(1, COLD, HEALTH_SUSPECT)
    assert status == HVAC_OFF
    assert reason == REASON_TEMP_SUSPECT


# --------------------------------------------------------------------------
# Single-output / no-conflict guarantees
# --------------------------------------------------------------------------
def test_heating_and_cooling_are_mutually_exclusive():
    """No temperature can produce both modes; asserted across the full band."""
    temperatures = [cfg.HVAC_HEATING_THRESHOLD - i * 0.1 for i in range(200)]
    for temperature in temperatures:
        status, _ = decide_hvac(1, temperature, HEALTH_NORMAL)
        assert status in (HVAC_COOLING, HVAC_HEATING, HVAC_OFF)
        assert not (status == HVAC_COOLING and status == HVAC_HEATING)


def test_control_decision_carries_both_reasons():
    decision = decide_control(1, HOT, DARK, HEALTH_NORMAL, HEALTH_NORMAL)
    assert decision.light_status == LIGHT_ON
    assert decision.hvac_status == HVAC_COOLING
    assert REASON_OCCUPIED_DARK in decision.control_reason
    assert REASON_COOLING in decision.control_reason


def test_suspect_sensors_override_both_actuators():
    decision = decide_control(1, HOT, DARK, HEALTH_SUSPECT, HEALTH_SUSPECT)
    assert decision.light_status == LIGHT_OFF
    assert decision.hvac_status == HVAC_OFF
    assert REASON_LIGHT_SUSPECT in decision.control_reason
    assert REASON_TEMP_SUSPECT in decision.control_reason


# --------------------------------------------------------------------------
# Flatline detector unit tests (synthetic, independent of the dataset)
# --------------------------------------------------------------------------
def test_flatline_flags_sustained_run_but_not_short_repeat():
    from detection import flatline_flags

    # 4 identical samples then a different value. Window 4, range 0.
    repeated = [21.0, 21.0, 21.0, 21.0, 22.0]
    flags = flatline_flags(np.asarray(repeated), window=4, max_range=0.0)
    # The confirmed window is samples 0-3, and a confirmed window marks all of
    # its own samples, so all four are flagged. Sample 4 differs and is not.
    assert flags.tolist() == [True, True, True, True, False]


def test_flatline_needs_the_full_window_before_firing():
    from detection import flatline_flags

    # Only three flat samples to start, so nothing is confirmed yet.
    flags = flatline_flags(np.asarray([21.0, 21.0, 21.0, 25.0]), 4, 0.0)
    assert not flags.any()


def test_flatline_respects_range_threshold():
    from detection import flatline_flags

    drifting = [21.0, 21.05, 21.1, 21.15, 21.2]   # range 0.2 over 5 samples
    strict = flatline_flags(np.asarray(drifting), 5, 0.1)
    loose = flatline_flags(np.asarray(drifting), 5, 0.5)
    assert not strict.any(), "range 0.2 exceeds the 0.1 threshold"
    assert loose.all(), "range 0.2 is within the 0.5 threshold"


def test_flatline_catches_the_jittered_pattern_of_a_real_stuck_sensor():
    """The injected anomalies dither by a bit rather than repeating exactly."""
    from detection import flatline_flags

    temperature_stuck = [26.75, 26.72, 26.73, 26.73, 26.73]
    light_stuck = [573, 574, 572, 574, 572, 574, 573]
    assert flatline_flags(np.asarray(temperature_stuck), 5, 0.03).all()
    assert flatline_flags(np.asarray(light_stuck), 7, 2.0).all()


def test_flatline_empty_and_single_sample_are_safe():
    from detection import flatline_flags

    assert not flatline_flags(np.asarray([]), 4, 0.0).any()
    assert not flatline_flags(np.asarray([1.0]), 4, 0.0).any()
    assert not flatline_flags(np.asarray([1.0, 1.0, 1.0]), 4, 0.0).any()



def test_motion_flatline_needs_context():
    """A long dark night of zero motion must NOT be reported as a stuck sensor."""
    from detection import detect

    quiet_night = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-05", periods=300, freq="5min"),
            "hour": 2,
            "motion": [0] * 300,
            "light_level": [0] * 300,
            "temperature": [20.0 + i * 0.01 for i in range(300)],
            "occupancy": [0] * 300,
            "is_anomaly": [0] * 300,
        }
    )
    result = detect(quiet_night)
    assert not result.motion_flatline.any(), "dark night misread as stuck motion"

    # The same run in a brightly lit room is a genuine fault signature.
    bright_day = quiet_night.copy()
    bright_day["light_level"] = 800
    result_bright = detect(bright_day)
    assert result_bright.motion_flatline.any()


def test_sensor_health_only_flagged_by_flatline_not_by_zscore():
    """A z-score hit alone is an environment event, not a broken sensor."""
    from detection import detect

    # A genuinely varying baseline with one extreme final reading. An earlier
    # version of this fixture used a constant 22.0 for every prior sample,
    # which is itself a 399-sample flatline and therefore correctly marked the
    # sensor suspect - the detector was right and the fixture was wrong.
    baseline = [22.0 + (i % 7) * 0.1 for i in range(399)]
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-05", periods=400, freq="5min"),
            "hour": [i % 24 for i in range(400)],
            "motion": [i % 2 for i in range(400)],
            "light_level": [(i * 7) % 400 for i in range(400)],
            "temperature": baseline + [45.0],  # one extreme spike at the end
            "occupancy": [0] * 400,
            "is_anomaly": [0] * 400,
        }
    )
    result = detect(frame)
    assert result.zscore_flag[-1], "spike should trip the z-score"
    assert not result.temperature_flatline[-1], "baseline must not read as flat"
    assert result.temperature_sensor_health[-1] == HEALTH_NORMAL, (
        "a spike is an environment event; only a flatline marks a sensor suspect"
    )
    assert result.anomaly_detected[-1]
    assert result.anomaly_reason[-1] == "rolling_zscore"
