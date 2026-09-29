"""
Phase 5 - anomaly detection: rolling Z-score (inherited) plus flatline rules.

`RollingZScore` is imported from `src/modeling.py` and used with the exact
parameters Phase 4 selected on the training split (window=288, k=2.0). Nothing
in Phase 4 is modified or re-tuned; the complement added here is the flatline
family, which targets a failure mode a z-score structurally cannot see.

Why a z-score alone is not enough
---------------------------------
A stuck sensor holds one value. Once the rolling window fills with repeats, the
local standard deviation collapses toward zero AND the deviation from the
baseline is zero as well, so the ratio is undefined or tiny and nothing is
flagged. Phase 4 measured this: the rolling detector found 18 of 33 anomalies
(54.5%) dataset-wide, missing every stuck run.

Why "unchanged" is not sufficient on its own
--------------------------------------------
Neither is the converse problem acceptable. Measured on this dataset, natural
(non-anomalous) identical-value runs reach:

    temperature  3 samples
    light_level  9 samples
    motion      49 samples

so flagging any repeat would fire constantly. A flatline rule therefore needs a
duration, a tolerance, and - for motion and for light - contextual corroboration,
because a dark room at 03:00 legitimately reports no motion and no light for
hours.

Detection is on the RANGE (max - min) across a trailing window rather than on
exact value repetition. The injected stuck anomalies are not exact copies:

    temperature  [26.75, 26.72, 26.73, 26.73, 26.73]   range 0.03 C
    light_level [573, 574, 572, 574, 572, 574, 573]   range 2 lux

An exact-equality rule therefore finds none of them, while a loose equality rule
fires on ordinary dark nights. Measured over the non-anomalous rows, natural
6-sample temperature windows reach a range of 0.03 C, and 7-sample light windows
of zero range are common only because the nights are dark. That is why the light
rule additionally requires a lit-level mean: a constant reading of 0 is genuine
darkness, not a fault. With that gate the light rule fires once instead of 18
times on this dataset.

The temperature case is the honest limit of a range-only rule. The fault's range
is exactly 0.03 C and an overnight natural window is also 0.03 C, so no threshold
separates them: below 0.03 the fault is invisible, at 0.03 a few genuinely quiet
night windows are also reported. The choice is to accept 7 false-positive rows
out of 2016 (0.35%) in exchange for detecting the fault at all, and to report
that cost rather than bury it.

Every threshold in `FlatlineConfig` was chosen by measuring the natural and
injected distributions on this dataset, and `sensitivity_table()` reprints the
sweep so the operating point can be re-checked rather than taken on trust. The
thresholds are tuned for THIS generator, whose stuck anomalies freeze a value
with only the last-bit jitter shown above. Real hardware drifts slowly instead of
freezing, which would need wider range thresholds and re-measured false-positive
counts.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg  # noqa: E402
from modeling import RollingZScore  # noqa: E402  (Phase 4, reused unchanged)

# Reasons, kept as constants so the CSV vocabulary cannot drift from the code.
REASON_NONE = "none"
REASON_ZSCORE = "rolling_zscore"
REASON_TEMP_FLAT = "temperature_flatline"
REASON_LIGHT_FLAT = "light_flatline"
REASON_MOTION_FLAT = "motion_flatline"

HEALTH_NORMAL = "normal"
HEALTH_SUSPECT = "suspect"

LIGHT_ON, LIGHT_OFF = "ON", "OFF"
HVAC_COOLING, HVAC_HEATING, HVAC_OFF = "COOLING", "HEATING", "OFF"


@dataclass(frozen=True)
class FlatlineConfig:
    """Flatline thresholds, in samples and in minutes at the 5-minute interval.

    Detection is on the RANGE (max - min) across a trailing window, not on
    exact value repetition. A genuinely stuck sensor dithered by a least
    significant bit or two, so the injected anomalies look like
    `[26.75, 26.72, 26.73, 26.73, 26.73]` and `[573, 574, 572, 574, 572, 574]`
    rather than exact copies. An exact-equality rule finds none of them, and a
    loose equality rule fires on ordinary dark nights.

    A run is marked retrospectively: once a window is confirmed flat, every
    sample belonging to that window is flagged, not just its last member. In a
    live deployment the same condition can only be confirmed once `window`
    samples have elapsed, which is the detection latency of this rule.
    """

    # Temperature: 6 samples = 30 min, range at most 0.03 C. The injected fault
    # is 5 samples, and a 5-sample window also detects it but doubles the
    # false positives (12 -> 7 at 6 samples, measured), because a longer flat
    # window is a stronger claim about the sensor. A tighter range threshold
    # is not an option: the fault's range is exactly 0.03, so anything below
    # that misses it entirely, and the overnight minimum is also 0.03.
    temp_window: int = 6
    temp_max_range: float = 0.03
    # Light: 7 samples = 35 min, range at most 2 lux, and a lit-level mean. The
    # mean gate is the single most important part of this rule: without it the
    # detector fires 18 times on ordinary dark nights instead of once.
    light_window: int = 7
    light_max_range: int = 2
    light_min_mean: int = cfg.LIGHT_THRESHOLD
    # Motion: 24 h. Natural motion runs reach 49 samples, so a short threshold
    # would flag ordinary quiet periods constantly.
    motion_window: int = 288
    motion_max_range: int = 0
    motion_zero_context_light: int = 500

    @property
    def temp_minutes(self) -> int:
        return self.temp_window * cfg.SAMPLING_INTERVAL_MINUTES

    @property
    def light_minutes(self) -> int:
        return self.light_window * cfg.SAMPLING_INTERVAL_MINUTES

    @property
    def motion_minutes(self) -> int:
        return self.motion_window * cfg.SAMPLING_INTERVAL_MINUTES


DEFAULT_FLATLINE = FlatlineConfig()

# Readings are rounded to 2 decimals, so an exact range of 0.03 C is stored as
# 0.030000000000001138. Comparisons therefore carry a tolerance far below the
# smallest possible real difference (0.01) so that "at most 0.03" means what it
# says instead of losing the boundary to binary floating point.
_RANGE_EPSILON = 1e-9


def flatline_flags(
    values: np.ndarray, window: int, max_range: float
) -> np.ndarray:
    """Flag samples belonging to any trailing window whose range <= `max_range`.

    Each confirmed window marks its own `window` samples, so a flat run is
    reported in full instead of only from the sample that completed it.
    """
    series = pd.Series(np.asarray(values, dtype=float))
    rolling = series.rolling(window, min_periods=window)
    flat = ((rolling.max() - rolling.min()) <= max_range + _RANGE_EPSILON).fillna(False)
    ok = flat.to_numpy()

    # Sample i belongs to a confirmed window ending at any t in [i, i+window-1].
    flags = np.zeros(len(ok), dtype=bool)
    for offset in range(window):
        if offset < len(ok):
            flags[: len(ok) - offset] |= ok[offset:]
    return flags



@dataclass
class DetectionResult:
    """Per-observation detection output. Columns line up with control_output.csv."""

    anomaly_detected: np.ndarray        # bool
    anomaly_reason: list[str]
    temperature_sensor_health: list[str]
    light_sensor_health: list[str]
    motion_sensor_health: list[str]
    zscore_flag: np.ndarray             # bool, from the inherited Phase 4 detector
    temperature_flatline: np.ndarray
    light_flatline: np.ndarray
    motion_flatline: np.ndarray

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "anomaly_detected": self.anomaly_detected,
                "anomaly_reason": self.anomaly_reason,
                "temperature_sensor_health": self.temperature_sensor_health,
                "light_sensor_health": self.light_sensor_health,
                "motion_sensor_health": self.motion_sensor_health,
                "zscore_flag": self.zscore_flag,
                "temperature_flatline": self.temperature_flatline,
                "light_flatline": self.light_flatline,
                "motion_flatline": self.motion_flatline,
            }
        )


def detect(
    df: pd.DataFrame,
    flatline: FlatlineConfig = DEFAULT_FLATLINE,
    zscore_window: int = 288,
    zscore_k: float = 2.0,
) -> DetectionResult:
    """Run the inherited rolling Z-score and the flatline rules over a frame.

    Reason priority is specific-before-generic: a flatline is a better
    diagnosis than "deviated from recent history", so a sample that trips both
    is reported as a flatline. Both flags stay available in the result so the
    overlap is auditable rather than lost.

    Sensor health is deliberately NOT raised by a Z-score hit. A temperature
    spike is far more likely to be a real hot period than a broken sensor, and
    the spec requires a sensor fault to be distinguishable from an environment.
    Only a flatline - a signature no real environment reproduces - marks a
    sensor suspect, and that is what drives the Phase 5 fail-safe.
    """
    zscore_detector = RollingZScore(window=zscore_window, threshold=zscore_k)
    zscore_flag = zscore_detector.predict(df[cfg.DATASET_COLUMNS[2:5]])

    temp_flat = flatline_flags(
        df["temperature"].to_numpy(),
        flatline.temp_window,
        flatline.temp_max_range,
    )

    # The light gate is the difference between a useful detector and one that
    # cries wolf every night: a constant reading of 0 is real darkness, so a
    # flatline only counts when the frozen level is at or above the lighting
    # threshold, i.e. the room reads as lit but the value never varies.
    light_flat_raw = flatline_flags(
        df["light_level"].to_numpy(),
        flatline.light_window,
        flatline.light_max_range,
    )
    light_mean = (
        df["light_level"]
        .rolling(flatline.light_window, min_periods=1)
        .mean()
        .to_numpy()
    )
    light_flat = light_flat_raw & (light_mean >= flatline.light_min_mean)

    # Motion needs corroboration, per the spec: a long run of zeros during a
    # dark night is correct behaviour, not a fault. A long run of ONES is not
    # plausible at any hour, so it is flagged on duration alone.
    motion_unchanged = flatline_flags(
        df["motion"].to_numpy(),
        flatline.motion_window,
        flatline.motion_max_range,
    )
    motion_on = df["motion"].to_numpy() == 1
    context_active = (
        df["light_level"].to_numpy() >= flatline.motion_zero_context_light
    )
    motion_flat = motion_unchanged & (motion_on | context_active)

    n = len(df)
    reason: list[str] = []
    temp_health: list[str] = []
    light_health: list[str] = []
    motion_health: list[str] = []
    anomaly = np.zeros(n, dtype=bool)

    for i in range(n):
        found = None
        if temp_flat[i]:
            found = REASON_TEMP_FLAT
        elif light_flat[i]:
            found = REASON_LIGHT_FLAT
        elif motion_flat[i]:
            found = REASON_MOTION_FLAT
        elif zscore_flag[i]:
            found = REASON_ZSCORE

        reason.append(found or REASON_NONE)
        anomaly[i] = found is not None
        temp_health.append(HEALTH_SUSPECT if temp_flat[i] else HEALTH_NORMAL)
        light_health.append(HEALTH_SUSPECT if light_flat[i] else HEALTH_NORMAL)
        motion_health.append(HEALTH_SUSPECT if motion_flat[i] else HEALTH_NORMAL)

    return DetectionResult(
        anomaly_detected=anomaly,
        anomaly_reason=reason,
        temperature_sensor_health=temp_health,
        light_sensor_health=light_health,
        motion_sensor_health=motion_health,
        zscore_flag=zscore_flag,
        temperature_flatline=temp_flat,
        light_flatline=light_flat,
        motion_flatline=motion_flat,
    )


def sensitivity_table(df: pd.DataFrame) -> list[list[str]]:
    """Sweep flatline window/range settings and report real false positives.

    A flagged sample only counts as a hit if it is an `is_anomaly` row and only
    counts as a false positive if it is not, so the two columns cannot be a
    restatement of one another. The range sweep is the important axis: the
    duration alone does not separate a stuck sensor from a quiet night, and the
    table makes that visible instead of leaving it asserted in a comment.
    """
    rows: list[list[str]] = []
    candidates = [
        ("temperature", "temperature", [4, 5, 6, 8], [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]),
        ("light_level", "light_level", [5, 7, 9, 12], [0.0, 2.0, 5.0, 10.0, 20.0]),
    ]
    for label, column, windows, ranges in candidates:
        values = df[column].to_numpy().astype(float)
        anomalous = df["is_anomaly"].to_numpy() == 1
        for window in windows:
            for max_range in ranges:
                flags = flatline_flags(values, window, max_range)
                if label == "light_level":
                    mean = (
                        df["light_level"]
                        .rolling(window, min_periods=1)
                        .mean()
                        .to_numpy()
                    )
                    flags = flags & (mean >= DEFAULT_FLATLINE.light_min_mean)
                false_positives = int((flags & ~anomalous).sum())
                caught = int((flags & anomalous).sum())
                if caught == 0 and false_positives == 0:
                    continue
                rows.append([
                    label,
                    f"{window} ({window * cfg.SAMPLING_INTERVAL_MINUTES} min)",
                    f"{max_range:g}",
                    str(caught),
                    str(false_positives),
                ])
    return rows
