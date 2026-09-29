"""
Phase 2C - Historical IoT sensor dataset generation.

Generates a reproducible, realistic 7-day history for the smart room that the
ESP32 firmware in src/main.cpp models:

    PIR motion  -> motion        (0/1)
    LDR         -> light_level   (normalised 0-1000, NOT calibrated lux)
    DS18B20     -> temperature   (degrees Celsius)
    label       -> occupancy     (0/1)
    ground truth-> is_anomaly    (0/1)

WHY A PYTHON SIMULATION INSTEAD OF WOKWI?
Wokwi is used to demonstrate the hardware/sensor layer. A 5-second firmware
interval would take ~2.8 days of real time to emit 2016 samples, so the ML
pipeline is trained on this simulated history instead. The two layers share
the same thresholds and the same normalised scales, so they stay consistent.

This module produces DATA ONLY. There is deliberately no machine learning
here: the Decision Tree, Linear Regression and anomaly detector belong to
later phases.

Run:  python src/data_generator.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow "python src/data_generator.py" as well as "import src.data_generator".
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg  # noqa: E402


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------
def generate_timestamps() -> pd.Series:
    """Chronological 5-minute timestamps covering the configured window."""
    return pd.Series(
        pd.date_range(
            start=cfg.START_TIMESTAMP,
            periods=cfg.EXPECTED_ROWS,
            freq=f"{cfg.SAMPLING_INTERVAL_MINUTES}min",
        )
    )


# ---------------------------------------------------------------------------
# Occupancy
# ---------------------------------------------------------------------------
# Probability that the room is occupied at a given hour on a normal weekday.
# Index 0 == hour 0. This is a PROBABILITY, not a rule: occupancy is sampled,
# so it is never a deterministic function of the hour.
WEEKDAY_OCCUPANCY_PROFILE = np.array([
    0.03, 0.02, 0.02, 0.02, 0.02, 0.04,   # 00:00-05:59  night
    0.12, 0.35, 0.72, 0.85, 0.88, 0.86,   # 06:00-11:59  morning ramp
    0.70, 0.78, 0.85, 0.86, 0.82, 0.72,   # 12:00-17:59  lunch dip, afternoon
    0.50, 0.32, 0.18, 0.10, 0.05, 0.04,   # 18:00-23:59  evening decline
])

WEEKEND_OCCUPANCY_FACTOR = 0.30
WEEKEND_OCCUPANCY_CAP = 0.35

# People arrive and leave in bursts, they do not teleport every 5 minutes, so
# occupancy is a two-state Markov chain. P(stay occupied) rises with the
# time-of-day probability, which produces realistic runs of occupied samples.
STAY_BASE = 0.70
STAY_SLOPE = 0.28
STAY_CAP = 0.95

# Per-day multiplier so no two days look identical.
DAY_VARIATION_MIN = 0.85
DAY_VARIATION_MAX = 1.15

OCCUPANCY_PROB_CAP = 0.95


def generate_occupancy(timestamps: pd.Series, rng: np.random.Generator) -> np.ndarray:
    """Sample occupancy using time-of-day probabilities plus day-to-day drift."""
    n = len(timestamps)
    hours = timestamps.dt.hour.to_numpy()
    days = timestamps.dt.dayofweek.to_numpy()      # 0=Mon .. 6=Sun
    dates = timestamps.dt.normalize().to_numpy()

    # One variation factor per calendar day, not per sample.
    unique_dates = pd.unique(dates)
    day_factor = {
        date: rng.uniform(DAY_VARIATION_MIN, DAY_VARIATION_MAX)
        for date in unique_dates
    }

    base = np.empty(n, dtype=float)
    for i in range(n):
        p = WEEKDAY_OCCUPANCY_PROFILE[hours[i]]

        if days[i] >= 5:  # Saturday or Sunday
            p = min(WEEKEND_OCCUPANCY_CAP, p * WEEKEND_OCCUPANCY_FACTOR)

        p *= day_factor[dates[i]]
        base[i] = min(OCCUPANCY_PROB_CAP, p)

    # Two-state Markov chain: base[i] is P(occupied) from an empty room, and
    # P(stay occupied) is derived from the same daily demand.
    stay = np.minimum(STAY_CAP, STAY_BASE + STAY_SLOPE * base)

    occupancy = np.zeros(n, dtype=int)
    state = 0
    for i in range(n):
        if state == 0:
            state = int(rng.random() < base[i])
        else:
            state = int(rng.random() < stay[i])
        occupancy[i] = state

    return occupancy


# ---------------------------------------------------------------------------
# Motion (PIR)
# ---------------------------------------------------------------------------
# Motion is correlated with occupancy but is NOT the same thing. People sit
# still, and sensors pick up stray movement in empty rooms. Keeping these
# probabilities well away from 0 and 1 is what stops a Decision Tree from
# simply learning "motion == occupancy".
MOTION_GIVEN_OCCUPIED = 0.65
MOTION_GIVEN_EMPTY = 0.05

# Motion arrives in bursts rather than flipping every 5 minutes.
MOTION_BURST_P = 0.55


def generate_motion(occupancy: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample the PIR channel, correlated with occupancy but independently noisy."""
    n = len(occupancy)
    base_p = np.where(
        occupancy == 1,
        MOTION_GIVEN_OCCUPIED,
        MOTION_GIVEN_EMPTY,
    )

    motion = np.zeros(n, dtype=int)
    state = 0
    for i in range(n):
        if state == 1:
            # Already moving: usually keeps moving, otherwise re-roll the base.
            p = max(base_p[i], MOTION_BURST_P)
        else:
            p = base_p[i]
        state = int(rng.random() < p)
        motion[i] = state

    return motion


# ---------------------------------------------------------------------------
# Light level (LDR)
# ---------------------------------------------------------------------------
# Daylight window, chosen so the simulated sun rises near 06:00 and sets near
# 18:00, matching a temperate climate.
SUNRISE_HOUR = 6.0
SUNSET_HOUR = 18.0

AMBIENT_FLOOR = 10.0     # residual indoor light at night
AMBIENT_PEAK = 680.0     # daylight contribution at solar noon

LIGHT_NOISE_SIGMA = 35.0
# Occupancy nudges the reading a little (movement, open blinds) but nowhere
# near enough for occupancy to be perfectly recoverable from light alone.
LIGHT_OCCUPANCY_BONUS = 25.0


def generate_light(
    timestamps: pd.Series,
    occupancy: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate the LDR as daylight + small occupancy effect + noise."""
    hours = timestamps.dt.hour.to_numpy() + timestamps.dt.minute.to_numpy() / 60.0

    # Half-sine daylight curve, zero outside the sunrise/sunset window.
    day_length = SUNSET_HOUR - SUNRISE_HOUR
    phase = (hours - SUNRISE_HOUR) / day_length
    daylight = np.where(
        (phase >= 0.0) & (phase <= 1.0),
        np.sin(phase * np.pi),
        0.0,
    )

    ambient = AMBIENT_FLOOR + AMBIENT_PEAK * daylight
    light = ambient + LIGHT_OCCUPANCY_BONUS * occupancy
    light += rng.normal(0.0, LIGHT_NOISE_SIGMA, size=len(light))

    return np.clip(light, cfg.LIGHT_MIN, cfg.LIGHT_MAX)


# ---------------------------------------------------------------------------
# Temperature (DS18B20)
# ---------------------------------------------------------------------------
TEMP_BASELINE = 23.5
# Daily cycle peaks mid-afternoon and bottoms out before dawn.
# Baseline 23.5 +/- 4.0 (+ up to 1.5 from occupant heat) puts normal
# temperatures in roughly 19.5-29.0 C. That deliberately straddles
# HVAC_COOLING_THRESHOLD (26 C) so the HVAC rule actually fires during busy
# afternoons instead of sitting at zero for the whole dataset.
TEMP_DAILY_AMPLITUDE = 4.0
TEMP_PEAK_HOUR = 15.0

# Occupants add heat, but the effect builds up and fades rather than switching
# instantly, so it is applied as a trailing moving average.
TEMP_OCCUPANCY_GAIN = 1.5
TEMP_OCCUPANCY_WINDOW = 12      # 12 samples == 1 hour

TEMP_NOISE_SIGMA = 0.15
# Light smoothing applied after the daily component, so adjacent samples stay
# close together exactly as a real slow-responding sensor behaves.
TEMP_SMOOTH_WINDOW = 3


def generate_temperature(
    timestamps: pd.Series,
    occupancy: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate a smooth indoor temperature with a daily cycle."""
    hours = timestamps.dt.hour.to_numpy() + timestamps.dt.minute.to_numpy() / 60.0

    daily = TEMP_DAILY_AMPLITUDE * np.cos(
        2.0 * np.pi * (hours - TEMP_PEAK_HOUR) / 24.0
    )

    # Trailing average of occupancy = heat that has accumulated recently.
    occupancy_heat = (
        pd.Series(occupancy)
        .rolling(window=TEMP_OCCUPANCY_WINDOW, min_periods=1)
        .mean()
        .to_numpy()
    )
    heat = TEMP_OCCUPANCY_GAIN * occupancy_heat

    temperature = TEMP_BASELINE + daily + heat
    temperature += rng.normal(0.0, TEMP_NOISE_SIGMA, size=len(temperature))

    temperature = pd.Series(temperature).rolling(
        window=TEMP_SMOOTH_WINDOW, min_periods=1
    ).mean().to_numpy()

    return np.clip(temperature, cfg.TEMP_MIN, cfg.TEMP_MAX)


# ---------------------------------------------------------------------------
# Anomaly injection
# ---------------------------------------------------------------------------
TEMP_SPIKE_MIN, TEMP_SPIKE_MAX = 10.0, 14.0
TEMP_DROP_MIN, TEMP_DROP_MAX = 12.0, 16.0
LIGHT_SPIKE_MIN, LIGHT_SPIKE_MAX = 950, 1000
LIGHT_DROP_MIN, LIGHT_DROP_MAX = 0, 30
# Frozen readings are not bit-identical, a real stuck ADC still dithers a little.
STUCK_SENSOR_JITTER = 0.01

# Do not start an anomaly event in the final slots, so stuck runs fit.
EVENT_MARGIN = 10


def _choose_anomaly_plan(
    n_rows: int, target_rows: int, rng: np.random.Generator
) -> list[tuple[str, int, int]]:
    """
    Build a non-overlapping list of (type, start, length) anomaly events.

    Each anomaly type is given its own row budget derived from
    ANOMALY_TYPE_WEIGHTS, and every type is guaranteed at least one event.
    Sampling types with replacement instead would let a low-weight type such
    as motion_false_trigger end up with zero events purely by chance, which
    silently removes a whole test case from the dataset.
    """
    types = list(cfg.ANOMALY_TYPE_WEIGHTS.keys())
    weights = np.array([cfg.ANOMALY_TYPE_WEIGHTS[t] for t in types], dtype=float)
    weights = weights / weights.sum()

    budgets = {t: int(round(target_rows * w)) for t, w in zip(types, weights)}
    for anomaly_type in types:
        budgets[anomaly_type] = max(1, budgets[anomaly_type])

    occupied = np.zeros(n_rows, dtype=bool)
    events: list[tuple[str, int, int]] = []
    rows_used = 0

    def find_free_start(length: int) -> int | None:
        """Random free position with EVENT_MARGIN slots kept clear at the end."""
        for _ in range(500):
            start = int(rng.integers(1, n_rows - length - EVENT_MARGIN))
            if not occupied[start:start + length].any():
                return start
        # Fall back to a linear scan so a crowded series still places the event.
        for start in range(1, n_rows - length - EVENT_MARGIN):
            if not occupied[start:start + length].any():
                return start
        return None

    # Pass 1: honour each type's budget.
    for anomaly_type in types:
        remaining = budgets[anomaly_type]
        while remaining > 0 and rows_used < target_rows:
            if anomaly_type == "stuck_sensor":
                # Never emit a run shorter than STUCK_SENSOR_MIN_LEN: a
                # two-sample "stuck" run is indistinguishable from normal
                # drift and is not a usable test case. If the remaining
                # budget cannot fit a full run, stop and let the top-up
                # pass spend those rows on single-sample anomalies.
                if remaining < cfg.STUCK_SENSOR_MIN_LEN:
                    break
                length = int(
                    rng.integers(
                        cfg.STUCK_SENSOR_MIN_LEN, cfg.STUCK_SENSOR_MAX_LEN + 1
                    )
                )
                length = min(length, remaining)
            else:
                length = 1

            start = find_free_start(length)
            if start is None:
                break

            occupied[start:start + length] = True
            events.append((anomaly_type, start, length))
            rows_used += length
            remaining -= length

    # Pass 2: top up to the target with extra single-row events.
    while rows_used < target_rows:
        anomaly_type = types[int(rng.choice(len(types), p=weights))]
        start = find_free_start(1)
        if start is None:
            break
        occupied[start] = True
        events.append((anomaly_type, start, 1))
        rows_used += 1

    return sorted(events, key=lambda event: event[1])


def inject_anomalies(
    df: pd.DataFrame,
    rng: np.random.Generator,
    return_plan: bool = False,
):
    """
    Corrupt a small number of samples and record the ground truth.

    This runs LAST, after smoothing, so that smoothing cannot quietly repair
    an injected fault. is_anomaly is set from what was actually corrupted and
    never from any statistical detector.

    When return_plan is True the injected events are returned alongside the
    frame. The plan is deliberately NOT written to the CSV, because the dataset
    schema is fixed at seven columns; it exists so tests and the final report
    can quote exact anomaly types instead of guessing them back out of values.
    """
    df = df.copy()
    df["is_anomaly"] = 0

    n_rows = len(df)
    target_rows = int(round(n_rows * cfg.ANOMALY_RATE))
    events = _choose_anomaly_plan(n_rows, target_rows, rng)

    temp = df["temperature"].to_numpy().astype(float).copy()
    light = df["light_level"].to_numpy().astype(float).copy()
    motion = df["motion"].to_numpy().astype(int).copy()
    flagged = np.zeros(n_rows, dtype=int)
    applied: list[tuple[str, str, int, int]] = []

    # Track which rows the plan already claimed, so an event can be relocated
    # without landing inside another event.
    planned = np.zeros(n_rows, dtype=bool)
    for _, start, length in events:
        planned[start:start + length] = True

    occupancy = df["occupancy"].to_numpy()

    # Alternate which channel a stuck run freezes. A coin flip here can easily
    # produce a dataset with no stuck temperature reading at all, even though
    # a frozen temperature is the classic stuck-sensor case.
    stuck_event_index = 0

    for anomaly_type, start, length in events:
        end = start + length
        # Events that relocate set these before being applied, so the row is
        # flagged at the position that was actually modified.
        flag_start, flag_len = start, length

        if anomaly_type == "temperature_spike":
            temp[start] += rng.uniform(TEMP_SPIKE_MIN, TEMP_SPIKE_MAX)
            applied.append((anomaly_type, "temperature", start, length))

        elif anomaly_type == "temperature_drop":
            temp[start] -= rng.uniform(TEMP_DROP_MIN, TEMP_DROP_MAX)
            applied.append((anomaly_type, "temperature", start, length))

        elif anomaly_type == "light_spike":
            light[start] = rng.uniform(LIGHT_SPIKE_MIN, LIGHT_SPIKE_MAX)
            applied.append((anomaly_type, "light_level", start, length))

        elif anomaly_type == "light_drop":
            light[start] = rng.uniform(LIGHT_DROP_MIN, LIGHT_DROP_MAX)
            applied.append((anomaly_type, "light_level", start, length))

        elif anomaly_type == "stuck_sensor":
            # Freeze on the last known-good value for the whole run.
            if start == 0:
                continue
            frozen_temp = temp[start - 1]
            frozen_light = light[start - 1]
            if stuck_event_index % 2 == 0:
                frozen_column = "temperature"
                temp[start:end] = frozen_temp + rng.normal(
                    0.0, STUCK_SENSOR_JITTER, size=length
                )
            else:
                frozen_column = "light_level"
                light[start:end] = frozen_light + rng.normal(
                    0.0, 1.0, size=length
                )
            stuck_event_index += 1
            applied.append((anomaly_type, frozen_column, start, length))

        elif anomaly_type == "motion_false_trigger":
            # A stray PIR detection in a room nobody is in. Force motion to 1
            # ONLY where it was genuinely 0 in an empty room: if motion was
            # already 1 the sample is not faulty at all, and flagging it would
            # create an "anomaly" that no detector could ever catch, which
            # would unfairly penalise the detector later.
            target = start
            limit = min(start + 30, n_rows)
            while target < limit and (
                planned[target] or motion[target] == 1 or occupancy[target] == 1
            ):
                target += 1
            if target >= limit:
                continue
            motion[target] = 1
            flag_start, flag_len = target, 1
            applied.append((anomaly_type, "motion", target, 1))

        else:  # pragma: no cover - guarded by the type table in config
            raise ValueError(f"Unknown anomaly type: {anomaly_type}")

        flagged[flag_start:flag_start + flag_len] = 1

    df["temperature"] = np.round(temp, 2)
    df["light_level"] = np.round(np.clip(light, cfg.LIGHT_MIN, cfg.LIGHT_MAX), 0).astype(int)
    df["motion"] = motion
    df["is_anomaly"] = flagged

    # Structural guarantee: every configured anomaly type must actually be
    # present, otherwise a whole test case silently disappears from the data.
    used_types = {entry[0] for entry in applied}
    missing = set(cfg.ANOMALY_TYPE_WEIGHTS) - used_types
    if missing:
        raise RuntimeError(f"Anomaly types not injected: {sorted(missing)}")

    plan = [
        (anomaly_type, column, start, length)
        for anomaly_type, column, start, length in applied
    ]

    if return_plan:
        return df, plan
    return df


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def generate_dataset(return_plan: bool = False):
    """Build the complete historical dataset."""
    rng = np.random.default_rng(cfg.SEED)

    timestamps = generate_timestamps()
    hours = timestamps.dt.hour

    occupancy = generate_occupancy(timestamps, rng)
    motion = generate_motion(occupancy, rng)
    light_level = generate_light(timestamps, occupancy, rng)
    temperature = generate_temperature(timestamps, occupancy, rng)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "hour": hours.astype(int),
        "motion": motion,
        "light_level": light_level.astype(int),
        "temperature": np.round(temperature, 2),
        "occupancy": occupancy,
    })

    result = inject_anomalies(df, rng, return_plan=return_plan)
    if return_plan:
        df, plan = result
    else:
        df = result

    df = df[cfg.DATASET_COLUMNS]
    if return_plan:
        return df, plan
    return df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_dataset(df: pd.DataFrame) -> list[tuple[str, bool, str]]:
    """
    Check the dataset against the Phase 2C acceptance criteria.

    Returns a list of (check_name, passed, detail) so nothing is asserted
    silently and every result can be printed.
    """
    checks: list[tuple[str, bool, str]] = []

    def add(name: str, passed: bool, detail: str = "") -> None:
        checks.append((name, bool(passed), detail))

    add(
        "Row count == 2016",
        len(df) == cfg.EXPECTED_ROWS,
        f"got {len(df)}, expected {cfg.EXPECTED_ROWS}",
    )

    add(
        "Columns match schema exactly",
        list(df.columns) == cfg.DATASET_COLUMNS,
        f"got {list(df.columns)}",
    )

    # Timestamps
    ts = pd.to_datetime(df["timestamp"])
    add("Timestamps chronological", bool(ts.is_monotonic_increasing), "")

    diffs = ts.diff().dropna().dt.total_seconds()
    expected_gap = cfg.SAMPLING_INTERVAL_MINUTES * 60
    add(
        f"Interval == {cfg.SAMPLING_INTERVAL_MINUTES} min",
        bool((diffs == expected_gap).all()),
        f"{diffs.min():.0f}-{diffs.max():.0f}s gaps seen",
    )

    add("No duplicate timestamps", bool(ts.duplicated().sum() == 0),
        f"{int(ts.duplicated().sum())} duplicates")

    # hour column consistency
    add(
        "hour matches timestamp",
        bool((ts.dt.hour.to_numpy() == df["hour"].to_numpy()).all()),
        "",
    )

    # Binary columns
    for column in ("motion", "occupancy", "is_anomaly"):
        values = set(df[column].unique().tolist())
        add(
            f"{column} is binary 0/1",
            values <= {0, 1},
            f"values={sorted(values)}",
        )

    # light_level
    add(
        f"light_level within {cfg.LIGHT_MIN}-{cfg.LIGHT_MAX}",
        bool(df["light_level"].between(cfg.LIGHT_MIN, cfg.LIGHT_MAX).all()),
        f"min={df['light_level'].min()} max={df['light_level'].max()}",
    )

    # temperature: normal rows must be in band, anomaly rows are allowed out
    normal_temp = df.loc[df["is_anomaly"] == 0, "temperature"]
    add(
        "temperature in band for NON-anomalous rows",
        bool(normal_temp.between(cfg.TEMP_MIN, cfg.TEMP_MAX).all()),
        f"min={normal_temp.min():.2f} max={normal_temp.max():.2f}",
    )

    # Anomaly rate
    rate = float(df["is_anomaly"].mean())
    add(
        "Anomaly rate within 1-3%",
        0.01 <= rate <= 0.03,
        f"{rate * 100:.2f}% ({int(df['is_anomaly'].sum())} rows)",
    )

    # Missing values
    missing = int(df.isna().sum().sum())
    add("No missing values", missing == 0, f"{missing} NaN cells")

    # Anti-leakage: motion must not be a copy of occupancy
    agreement = float((df["motion"] == df["occupancy"]).mean())
    add(
        "motion is not identical to occupancy",
        agreement < 1.0,
        f"motion==occupancy on {agreement * 100:.1f}% of rows",
    )

    # Anti-leakage: occupancy must not be a deterministic function of hour.
    # If it were, every hour-of-day would map to a single value and
    # `mixed_hours` would be 0. Measuring the *count* of mixed hours is the
    # honest test; a bare `.max() == 2` would pass even if 23 of 24 hours
    # were constant, which is not the property we care about.
    per_hour = df.groupby("hour")["occupancy"].nunique()
    mixed_hours = int((per_hour == 2).sum())
    homogeneous = [int(h) for h in per_hour.index[per_hour == 1]]
    add(
        "occupancy is not deterministic given hour",
        mixed_hours / len(per_hour) >= 0.8,
        (
            f"{mixed_hours}/{len(per_hour)} hours contain both classes"
            + (f"; single-class hours: {homogeneous}" if homogeneous else "")
        ),
    )

    # The dataset is useless for the control-logic phase if the two control
    # rules never fire, so assert that both are actually exercised.
    clean = df["is_anomaly"] == 0

    lighting = (df["occupancy"] == 1) & clean & (df["light_level"] < cfg.LIGHT_THRESHOLD)
    hvac = (
        (df["occupancy"] == 1)
        & clean
        & (df["temperature"] > cfg.HVAC_COOLING_THRESHOLD)
    )
    add(
        "Lighting rule activates on real rows",
        bool(lighting.sum() > 0),
        f"{int(lighting.sum())} rows ({lighting.mean() * 100:.1f}%)",
    )
    add(
        "HVAC rule activates on real rows",
        bool(hvac.sum() > 0),
        f"{int(hvac.sum())} rows ({hvac.mean() * 100:.1f}%)",
    )

    return checks


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_summary(df: pd.DataFrame) -> None:
    """Print statistics computed from the generated data. Nothing is hard-coded."""
    ts = pd.to_datetime(df["timestamp"])
    normal = df.loc[df["is_anomaly"] == 0]

    print()
    print("Dataset generated successfully")
    print("=" * 46)
    print(f"Rows            : {len(df)}")
    print(f"Columns         : {len(df.columns)}  ({', '.join(df.columns)})")
    print(f"Date range      : {ts.min()} -> {ts.max()}")
    print(f"Seed            : {cfg.SEED}")
    print()
    print("Occupancy:")
    print(f"  Occupied      : {int(df['occupancy'].sum())} "
          f"({df['occupancy'].mean() * 100:.1f}%)")
    print(f"  Unoccupied    : {int((df['occupancy'] == 0).sum())} "
          f"({(df['occupancy'] == 0).mean() * 100:.1f}%)")
    print()
    print("Motion:")
    print(f"  Motion        : {int(df['motion'].sum())} "
          f"({df['motion'].mean() * 100:.1f}%)")
    print(f"  No motion     : {int((df['motion'] == 0).sum())}")
    print()
    print("Light (normalised 0-1000 scale, not calibrated lux):")
    print(f"  Min / Max     : {df['light_level'].min()} / {df['light_level'].max()}")
    print(f"  Mean          : {df['light_level'].mean():.1f}")
    print()
    print("Temperature:")
    print(f"  Min / Max     : {df['temperature'].min():.2f} / {df['temperature'].max():.2f}")
    print(f"  Mean          : {df['temperature'].mean():.2f}")
    print(f"  Min / Max (non-anomalous): "
          f"{normal['temperature'].min():.2f} / {normal['temperature'].max():.2f}")
    print()
    print("Anomalies:")
    print(f"  Count         : {int(df['is_anomaly'].sum())}")
    print(f"  Rate          : {df['is_anomaly'].mean() * 100:.2f}%")
    print()
    print(f"Missing values  : {int(df.isna().sum().sum())}")


def print_validation(checks: list[tuple[str, bool, str]]) -> bool:
    print()
    print("Validation")
    print("=" * 46)
    for name, passed, detail in checks:
        mark = "PASS" if passed else "FAIL"
        suffix = f"  ({detail})" if detail else ""
        print(f"  [{mark}] {name}{suffix}")
    all_passed = all(passed for _, passed, _ in checks)
    print()
    print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
    return all_passed


# ---------------------------------------------------------------------------
# Optional sanity plot
# ---------------------------------------------------------------------------
def save_overview_plot(df: pd.DataFrame) -> Path:
    """
    Minimal Phase 2C sanity plot. Deliberately plain - full EDA is Phase 3.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg.VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)

    ts = pd.to_datetime(df["timestamp"])
    colors = np.where(df["is_anomaly"] == 1, "tab:red", "tab:blue")

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

    axes[0].plot(ts, df["temperature"], lw=0.7, color="tab:green", label="temperature")
    axes[0].scatter(
        ts[df["is_anomaly"] == 1],
        df.loc[df["is_anomaly"] == 1, "temperature"],
        s=12, color="tab:red", label="injected anomaly", zorder=3,
    )
    axes[0].set_ylabel("Temperature (C)")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)

    axes[1].plot(ts, df["light_level"], lw=0.7, color="tab:orange", label="light_level")
    axes[1].scatter(
        ts[df["is_anomaly"] == 1],
        df.loc[df["is_anomaly"] == 1, "light_level"],
        s=12, color="tab:red", zorder=3,
    )
    axes[1].set_ylabel("Light level (0-1000)")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)

    axes[2].plot(ts, df["occupancy"], lw=1.0, color="tab:purple",
                 label="occupancy (ground truth)")
    axes[2].plot(ts, df["motion"], lw=0.6, color=colors[0], alpha=0.7,
                 label="motion (PIR)")
    axes[2].set_ylabel("Binary")
    axes[2].set_xlabel("Time")
    axes[2].legend(loc="upper right")
    axes[2].grid(alpha=0.3)

    fig.suptitle(
        f"Raw simulated sensor overview - {len(df)} samples, "
        f"{cfg.DATASET_DAYS} days @ {cfg.SAMPLING_INTERVAL_MINUTES} min, seed {cfg.SEED}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(cfg.OVERVIEW_PNG, dpi=110)
    plt.close(fig)

    return cfg.OVERVIEW_PNG


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    df = generate_dataset()

    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.DATASET_CSV, index=False)

    print_summary(df)
    all_passed = print_validation(validate_dataset(df))

    plot_path = save_overview_plot(df)
    print(f"CSV written     : {cfg.DATASET_CSV}")
    print(f"Plot written    : {plot_path}")
    print()

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
