"""
Phase 5 - deterministic rule-based control with fail-safe behaviour.

This module is pure logic: no I/O, no model, no randomness. Every decision is a
total function of (predicted occupancy, temperature, light level, sensor
health), so the truth-table tests in tests/test_control_logic.py are exact
rather than statistical.

Fail-safe policy
----------------
When a sensor that a decision depends on is `suspect`, the corresponding
actuator is forced OFF and the reason records which sensor was distrusted. The
measurement is never silently replaced with a plausible-looking number, and the
rule is deliberately conservative: an untrustworthy reading is treated as
"insufficient evidence to act", not as evidence for the opposite action.

`hvac_status` is a single three-valued field (OFF / COOLING / HEATING) rather
than two independent booleans, so heating and cooling cannot both be asserted.
The thresholds are mutually exclusive (26.0 C and 20.0 C) and that exclusivity
is asserted at import time below.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

import config as cfg
from detection import (
    HEALTH_NORMAL,
    HEALTH_SUSPECT,
    HVAC_COOLING,
    HVAC_HEATING,
    HVAC_OFF,
    LIGHT_OFF,
    LIGHT_ON,
)

# Control reasons, namespaced so a reader can tell which actuator a reason
# refers to and whether it was a fail-safe or an ordinary decision.
REASON_OCCUPIED_DARK = "light_occupied_and_dark"
REASON_OCCUPIED_BRIGHT = "light_occupied_but_bright"
REASON_UNOCCUPIED = "light_unoccupied"
REASON_LIGHT_SUSPECT = "light_sensor_suspect"

REASON_COOLING = "hvac_occupied_and_hot"
REASON_HEATING = "hvac_occupied_and_cold"
REASON_WITHIN_BAND = "hvac_occupied_within_comfort_band"
REASON_UNOCCUPIED_HVAC = "hvac_unoccupied"
REASON_TEMP_SUSPECT = "temperature_sensor_suspect"

# The two HVAC modes must never be simultaneously satisfiable.
assert cfg.HVAC_COOLING_THRESHOLD > cfg.HVAC_HEATING_THRESHOLD, (
    "cooling threshold must exceed heating threshold, otherwise the two "
    "branches could both fire and produce conflicting outputs"
)


@dataclass(frozen=True)
class ControlDecision:
    light_status: str
    hvac_status: str
    light_reason: str
    hvac_reason: str
    control_reason: str


def decide_lighting(
    predicted_occupancy: int,
    light_level: float,
    light_sensor_health: str = HEALTH_NORMAL,
) -> tuple[str, str]:
    """Return (status, reason). Lighting ON only when occupied AND dark AND trusted.

    Strict `<` on the threshold, so a reading exactly at LIGHT_THRESHOLD is
    treated as sufficient light and the lamp stays off.
    """
    if light_sensor_health == HEALTH_SUSPECT:
        return LIGHT_OFF, REASON_LIGHT_SUSPECT
    if predicted_occupancy == 1:
        if light_level < cfg.LIGHT_THRESHOLD:
            return LIGHT_ON, REASON_OCCUPIED_DARK
        return LIGHT_OFF, REASON_OCCUPIED_BRIGHT
    return LIGHT_OFF, REASON_UNOCCUPIED


def decide_hvac(
    predicted_occupancy: int,
    temperature: float,
    temperature_sensor_health: str = HEALTH_NORMAL,
) -> tuple[str, str]:
    """Return (status, reason) with a single three-valued HVAC output.

    Fail-safe first: an untrusted temperature stops the HVAC entirely rather
    than defaulting to a mode based on a reading we do not believe.
    """
    if temperature_sensor_health == HEALTH_SUSPECT:
        return HVAC_OFF, REASON_TEMP_SUSPECT
    if predicted_occupancy != 1:
        return HVAC_OFF, REASON_UNOCCUPIED_HVAC
    if temperature > cfg.HVAC_COOLING_THRESHOLD:
        return HVAC_COOLING, REASON_COOLING
    if temperature < cfg.HVAC_HEATING_THRESHOLD:
        return HVAC_HEATING, REASON_HEATING
    return HVAC_OFF, REASON_WITHIN_BAND


def decide_control(
    predicted_occupancy: int,
    temperature: float,
    light_level: float,
    temperature_sensor_health: str = HEALTH_NORMAL,
    light_sensor_health: str = HEALTH_NORMAL,
) -> ControlDecision:
    """Evaluate both actuators and return a single, fully attributed decision."""
    light_status, light_reason = decide_lighting(
        predicted_occupancy, light_level, light_sensor_health
    )
    hvac_status, hvac_reason = decide_hvac(
        predicted_occupancy, temperature, temperature_sensor_health
    )
    return ControlDecision(
        light_status=light_status,
        hvac_status=hvac_status,
        light_reason=light_reason,
        hvac_reason=hvac_reason,
        control_reason=f"{light_reason}; {hvac_reason}",
    )


def decide_all(
    predicted_occupancy,
    temperature,
    light_level,
    temperature_sensor_health,
    light_sensor_health,
) -> "pd.DataFrame":
    """Vectorised convenience wrapper returning every decision column.

    The replay is written against this, which in turn calls the same
    `decide_control` the tests exercise, so the historical output cannot drift
    away from the tested logic. Both statuses and both reasons are returned;
    returning only one of them would invite the replay to silently drop half
    the control output.
    """
    decisions = [
        decide_control(int(o), float(t), float(l), th, lh)
        for o, t, l, th, lh in zip(
            predicted_occupancy,
            temperature,
            light_level,
            temperature_sensor_health,
            light_sensor_health,
        )
    ]
    return pd.DataFrame(
        {
            "light_status": [d.light_status for d in decisions],
            "hvac_status": [d.hvac_status for d in decisions],
            "light_reason": [d.light_reason for d in decisions],
            "hvac_reason": [d.hvac_reason for d in decisions],
            "control_reason": [d.control_reason for d in decisions],
        }
    )
