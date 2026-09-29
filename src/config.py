"""
Central configuration for the historical IoT dataset generator.

These values deliberately mirror the ESP32 firmware (src/main.cpp) so the
simulated history and the real hardware layer describe the same room:

  LIGHT_THRESHOLD          == LIGHT_THRESHOLD        in src/main.cpp
  HVAC_COOLING_THRESHOLD   == HVAC_COOLING_THRESHOLD in src/main.cpp
  SENSOR_INTERVAL_MINUTES  == SENSOR_INTERVAL / 60000 in src/main.cpp

If a control threshold changes in the firmware, change it here too.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
# Every run of data_generator.py must produce an identical CSV (Rule 4).
SEED = 42

# --------------------------------------------------------------------------
# Dataset shape
# --------------------------------------------------------------------------
# 7 days at a 5-minute interval = 7 * 24 * 12 = 2016 rows.
# 5 minutes matches the ESP32 SENSOR_INTERVAL.
DATASET_DAYS = 7
SAMPLING_INTERVAL_MINUTES = 5

# A Monday, so the week contains 5 weekdays + 1 weekend day pair, which gives
# the occupancy model a realistic weekday/weekend difference.
START_TIMESTAMP = "2026-01-05 00:00:00"

EXPECTED_ROWS = DATASET_DAYS * 24 * (60 // SAMPLING_INTERVAL_MINUTES)

# --------------------------------------------------------------------------
# Sensor ranges
# --------------------------------------------------------------------------
# light_level is a NORMALISED 0-1000 scale. It is NOT calibrated physical lux.
# The ESP32 produces the same scale via map(analogRead(34), 0, 4095, 0, 1000).
LIGHT_MIN = 0
LIGHT_MAX = 1000

# Plausible indoor temperature band for normal (non-anomalous) samples.
TEMP_MIN = 18.0
TEMP_MAX = 32.0

# --------------------------------------------------------------------------
# Control thresholds (must match src/main.cpp)
# --------------------------------------------------------------------------
LIGHT_THRESHOLD = 300
HVAC_COOLING_THRESHOLD = 26.0
# Heating IS implemented in the Python control layer (control_logic.py), which
# returns a three-valued OFF / COOLING / HEATING field. The ESP32 firmware still
# only has a cooling branch and reports a single boolean hvacOn, so heating is
# not yet expressible on the wire. See docs/LIMITATIONS.md.
HVAC_HEATING_THRESHOLD = 20.0

# --------------------------------------------------------------------------
# Anomaly injection
# --------------------------------------------------------------------------
# Fraction of rows that receive an injected, known-bad reading.
# Spec target: 1-3%. 2% of 2016 rows = ~40 rows.
ANOMALY_RATE = 0.02

# Relative mix of anomaly types. These are weights, not probabilities.
# The generator turns these into per-type row budgets and guarantees every
# type is present at least once, so a low weight still yields a test case.
ANOMALY_TYPE_WEIGHTS = {
    "temperature_spike": 0.18,
    "temperature_drop": 0.17,
    "light_spike": 0.14,
    "light_drop": 0.11,
    "stuck_sensor": 0.30,
    "motion_false_trigger": 0.10,
}

# Stuck-sensor anomalies hold a value steady across this many samples.
STUCK_SENSOR_MIN_LEN = 4
STUCK_SENSOR_MAX_LEN = 8

# --------------------------------------------------------------------------
# Output paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
VISUALIZATIONS_DIR = PROJECT_ROOT / "visualizations"

DATASET_CSV = DATA_DIR / "sensor_data.csv"
OVERVIEW_PNG = VISUALIZATIONS_DIR / "raw_sensor_overview.png"

# --------------------------------------------------------------------------
# Output schema
# --------------------------------------------------------------------------
DATASET_COLUMNS = [
    "timestamp",
    "hour",
    "motion",
    "light_level",
    "temperature",
    "occupancy",
    "is_anomaly",
]
