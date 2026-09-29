"""
Phase 3 - preprocessing utilities for the occupancy / temperature models.

This module is deliberately small. It provides the reusable primitives that
Phase 4 (model training) will need, and nothing else:

    load_dataset            read the CSV, parse timestamps, sort chronologically
    validate_dataset        data-quality checks, no silent repair
    chronological_split     time-based train/test split (NOT random)
    build_feature_matrix    assemble X / y for a given prediction task
    leakage_checks          prove the split and features cannot leak
    autocorrelation_report measure temporal structure of a column

Design rules enforced here rather than merely documented:

  1. `train_test_split` is never imported. The data is a time series, and a
     random split puts neighbouring 5-minute samples on both sides of the
     boundary, which inflates scores for models that exploit autocorrelation
     (temperature has lag-1 autocorrelation ~0.999).

  2. `is_anomaly` and `timestamp` are never model inputs. `is_anomaly` is
     ground truth recorded at injection time; feeding it to a model would
     leak the answer. `_assert_valid_features` runs at import time, so an
     unsafe feature list is a hard error, not a code-review comment.

  3. No scaler. A Decision Tree is invariant to feature scale, and for this
     feature set (hour 0-23, light 0-1000, temperature ~18-32) linear
     regression is well conditioned. Introducing StandardScaler "because
     tutorials do" would be cargo cult. If later experiments need it, add it
     in Phase 4 with a stated reason.

  4. The source CSV is read-only. Nothing in this module writes to disk.

Source data: data/sensor_data.csv, produced and validated in Phase 2C.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the Phase 2C configuration instead of duplicating constants, so the
# dataset shape and the analysis can never drift apart.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg  # noqa: E402


# --------------------------------------------------------------------------
# Feature / target definitions for the three later models
# --------------------------------------------------------------------------
# Rationale for each column is in the Phase 3 report; the short version:

#   hour         time-of-day prior. People arrive and leave on a schedule, so
#                hour carries real signal on its own. Kept despite two
#                single-class hours (2 and 4) - dropping it would throw away
#                22 genuinely informative hours.
#   motion       PIR. The strongest single predictor: occupied rooms are
#                usually emitting motion.
#   light_level  LDR. Distinguishes "lit, occupied" from "dark, occupied"
#                (early morning, evening). Implied by occupancy-driven
#                lighting, so it is a *proxy*, not independent evidence.
#   temperature  DS18B20. Occupants and equipment add heat. Strong causal
#                link, but also the most autocorrelated column in the set.

OCCUPANCY_FEATURES = ["hour", "motion", "light_level", "temperature"]
OCCUPANCY_TARGET = "occupancy"

TEMPERATURE_FEATURES = ["hour", "motion", "light_level", "occupancy"]
TEMPERATURE_TARGET = "temperature"

# Anomaly detection treats all three sensors as evidence and is_anomaly as the
# label to be recovered. is_anomaly is an output here, never an input.
ANOMALY_FEATURES = ["motion", "light_level", "temperature"]
ANOMALY_TARGET = "is_anomaly"

TASKS = {
    "occupancy": (OCCUPANCY_FEATURES, OCCUPANCY_TARGET),
    "temperature": (TEMPERATURE_FEATURES, TEMPERATURE_TARGET),
    "anomaly": (ANOMALY_FEATURES, ANOMALY_TARGET),
}

# Never allowed as a model input, for any task.
FORBIDDEN_INPUTS = frozenset({"timestamp", "is_anomaly"})

# Chronological split: first 5 of 7 days train, last 2 days test.
TRAIN_DAYS = 5
TEST_DAYS = 2

# 288 samples per day at a 5-minute interval. Used for lag interpretation.
SAMPLES_PER_HOUR = 60 // cfg.SAMPLING_INTERVAL_MINUTES  # 12
SAMPLES_PER_DAY = 24 * SAMPLES_PER_HOUR  # 288


@dataclass
class Check:
    """One validation or leakage check, with the evidence that decided it."""

    name: str
    passed: bool
    detail: str

    def render(self) -> str:
        return f"[{'PASS' if self.passed else 'FAIL'}] {self.name}  ({self.detail})"


# --------------------------------------------------------------------------
# Import-time feature safety net
# --------------------------------------------------------------------------
def _assert_valid_features(features: list[str], target: str) -> None:
    """Reject a feature set that could leak. Called once per task at import."""
    if target in features:
        raise ValueError(f"target '{target}' present in its own feature list")
    leaked = FORBIDDEN_INPUTS.intersection(features)
    if leaked:
        raise ValueError(f"forbidden input column(s) in features: {sorted(leaked)}")
    unknown = [c for c in features if c not in cfg.DATASET_COLUMNS]
    if unknown:
        raise ValueError(f"feature(s) not present in the dataset: {unknown}")


for _task, (_feats, _tgt) in TASKS.items():
    _assert_valid_features(_feats, _tgt)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_dataset(path: Path | str | None = None) -> pd.DataFrame:
    """Load the dataset, parse `timestamp` and sort chronologically.

    Sorts even though the file is already ordered, so that callers can rely on
    the ordering invariant regardless of how the file was written.
    """
    path = Path(path) if path is not None else cfg.DATASET_CSV
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return add_hour_column(df)


def add_hour_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure `hour` is present and consistent with `timestamp`.

    Compares values rather than using Series.equals, which would fail purely
    on dtype: the CSV stores `hour` as int64 while `.dt.hour` yields int32.
    """
    df = df.copy()
    if "hour" not in df.columns:
        df["hour"] = df["timestamp"].dt.hour
        return df

    derived = df["timestamp"].dt.hour.to_numpy()
    stored = df["hour"].to_numpy()
    mismatched = int((derived != stored).sum())
    if mismatched:
        raise ValueError(
            f"'hour' column disagrees with the timestamp in {mismatched} rows"
        )
    return df


# --------------------------------------------------------------------------
# Data quality
# --------------------------------------------------------------------------
def validate_dataset(df: pd.DataFrame) -> list[Check]:
    """Verify structure and value sanity. Reports problems, never repairs them.

    Returns a list of Check. Callers decide how loudly to complain; the point
    is that a bad dataset surfaces as an explicit FAIL rather than as silently
    dropped rows downstream.
    """
    checks: list[Check] = []

    expected_cols = list(cfg.DATASET_COLUMNS)
    checks.append(
        Check(
            "schema matches Phase 2C contract",
            list(df.columns) == expected_cols,
            f"{len(df.columns)} cols"
            + ("" if list(df.columns) == expected_cols else f", expected {expected_cols}"),
        )
    )

    n_expected = cfg.EXPECTED_ROWS
    checks.append(
        Check(
            f"row count == {n_expected}",
            len(df) == n_expected,
            f"{len(df)} rows",
        )
    )

    dupes = int(df["timestamp"].duplicated().sum())
    checks.append(
        Check("no duplicate timestamps", dupes == 0, f"{dupes} duplicates")
    )

    deltas = df["timestamp"].diff().dropna()
    unique_deltas = sorted({d.total_seconds() for d in deltas})
    expected_step = cfg.SAMPLING_INTERVAL_MINUTES * 60
    uniform = (
        len(unique_deltas) == 1
        and unique_deltas[0] == expected_step
        and len(deltas) == len(df) - 1
    )
    checks.append(
        Check(
            f"uniform {expected_step // 60}-minute interval",
            uniform,
            f"observed steps: {[int(s) // 60 for s in unique_deltas]} min",
        )
    )

    missing_cells = int(df.isna().sum().sum())
    missing_cols = {c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().any()}
    checks.append(
        Check(
            "no missing values",
            missing_cells == 0,
            f"{missing_cells} NaN cells" + (f" in {missing_cols}" if missing_cols else ""),
        )
    )

    binary = ["motion", "occupancy", "is_anomaly"]
    bad_binary = {
        c: sorted(set(df[c].unique()) - {0, 1}) for c in binary
    }
    bad_binary = {c: v for c, v in bad_binary.items() if v}
    checks.append(
        Check(
            "binary columns contain only 0/1",
            not bad_binary,
            "all clean" if not bad_binary else f"unexpected values: {bad_binary}",
        )
    )

    out_of_range = int(
        ((df["light_level"] < cfg.LIGHT_MIN) | (df["light_level"] > cfg.LIGHT_MAX)).sum()
    )
    checks.append(
        Check(
            f"light_level within [{cfg.LIGHT_MIN}, {cfg.LIGHT_MAX}]",
            out_of_range == 0,
            f"{out_of_range} rows outside range",
        )
    )

    # Anomalies are expected to sit outside the plausible band, so the
    # plausibility check is reported separately over normal rows only.
    normal = df[df["is_anomaly"] == 0]
    temp_oob = int(
        ((normal["temperature"] < cfg.TEMP_MIN) | (normal["temperature"] > cfg.TEMP_MAX)).sum()
    )
    checks.append(
        Check(
            f"temperature within [{cfg.TEMP_MIN}, {cfg.TEMP_MAX}] on normal rows",
            temp_oob == 0,
            f"{temp_oob} normal rows outside band",
        )
    )

    rate = float(df["is_anomaly"].mean())
    checks.append(
        Check(
            "anomaly rate within 1-3%",
            0.01 <= rate <= 0.03,
            f"{rate * 100:.2f}% ({int(df['is_anomaly'].sum())} rows)",
        )
    )

    return checks


# --------------------------------------------------------------------------
# Chronological split
# --------------------------------------------------------------------------
def chronological_split(
    df: pd.DataFrame, train_days: int = TRAIN_DAYS, test_days: int = TEST_DAYS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by timestamp, earliest rows to train, later rows to test.

    Deliberately NOT `sklearn.model_selection.train_test_split`. Measured
    temperature lag-1 autocorrelation on the shipped CSV is 0.9106, so a
    random split hands the model its own answer: the test sample's immediate
    neighbour sits in the training set. Here the boundary is a single
    timestamp, so max(train) < min(test) holds by construction and no future
    observation is ever visible during training.

    The returned frames keep the original dataset index so predictions can be
    scored against the exact source rows.
    """
    n_train = train_days * SAMPLES_PER_DAY
    if n_train >= len(df):
        raise ValueError(
            f"train_days={train_days} needs {n_train} rows, dataset has {len(df)}"
        )

    cutoff = df["timestamp"].iloc[n_train - 1]
    train = df[df["timestamp"] <= cutoff].copy()
    test = df[df["timestamp"] > cutoff].copy()

    expected_test = test_days * SAMPLES_PER_DAY
    if len(test) != expected_test:
        raise ValueError(
            f"split produced {len(test)} test rows, expected {expected_test}"
        )
    return train, test


def cross_boundary_twin_mask(
    train: pd.DataFrame, test: pd.DataFrame, features: list[str] | None = None
) -> np.ndarray:
    """Boolean mask over `test` rows whose exact feature vector also occurs in `train`.

    Returns the mask, and the number of twin groups whose labels contradict
    each other. Used both by the leakage check and by the Phase 4 sensitivity
    ablation, so the two can never disagree about which rows are twins.
    """
    features = features or OCCUPANCY_FEATURES
    train_keys = pd.MultiIndex.from_frame(train[features])
    test_keys = pd.MultiIndex.from_frame(test[features])
    is_twin = test_keys.isin(train_keys)

    combined = pd.concat(
        [train.assign(_split="train"), test.assign(_split="test")],
        ignore_index=True,
    )
    dup_rows = combined[combined.duplicated(subset=features, keep=False)]
    conflicting = 0
    cross_groups = 0
    if len(dup_rows):
        for _, g in dup_rows.groupby(features, dropna=False):
            if g["_split"].nunique() > 1:
                cross_groups += 1
                if g["occupancy"].nunique() > 1:
                    conflicting += 1
    return is_twin, cross_groups, conflicting


def weekday_holdout_split(
    df: pd.DataFrame, holdout_weekday: str = "Wednesday"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """DIAGNOSTIC split: train on every day except one weekday, test on it.

    Purpose: separate "the model is weak" from "the test period is entirely
    weekend". The primary chronological split trains on zero weekend days, so
    a poor score there conflates distribution shift with model quality. This
    split keeps weekends in training and removes weekday coverage instead.

    Honesty about the trade-off: this is NOT chronologically pure. Training
    contains days that occur *after* the holdout day, so future information
    enters training. That is acceptable for a diagnostic and unacceptable for
    the headline result, which is why `chronological_split` remains primary.

    The default holdout is Wednesday (the middle weekday) rather than Monday,
    so the removed day is not adjacent to a dataset boundary.

    The returned frames deliberately KEEP the original dataset index rather
    than resetting it, so any prediction computed on them can be traced back
    to (and scored against) the exact source rows. Resetting the index here
    previously caused a weekday-holdout evaluation to score the wrong day.
    """
    names = df["timestamp"].dt.day_name()
    test = df[names == holdout_weekday].copy()
    train = df[names != holdout_weekday].copy()
    if test.empty or train.empty:
        raise ValueError(f"holdout_weekday='{holdout_weekday}' not present in data")
    return train, test


# --------------------------------------------------------------------------
# Feature assembly
# --------------------------------------------------------------------------
def build_feature_matrix(
    df: pd.DataFrame, task: str
) -> tuple[pd.DataFrame, pd.Series, list[str], str]:
    """Return (X, y, feature_names, target_name) for a named task.

    The index of X and y is preserved so callers can join them back to the
    original timestamps and audit which rows were used.
    """
    if task not in TASKS:
        raise KeyError(f"unknown task '{task}', expected one of {sorted(TASKS)}")
    features, target = TASKS[task]
    _assert_valid_features(features, target)
    return df[features].copy(), df[target].copy(), list(features), target


def hourly_occupancy_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Occupancy rate per hour-of-day, plus the mixed/single-class flag.

    The Phase 2C validation requires most hours to contain both classes. This
    returns the per-hour detail so Phase 4 can see exactly which hours carry
    no signal instead of trusting a single boolean.
    """
    grouped = df.groupby("hour")["occupancy"]
    profile = pd.DataFrame(
        {
            "samples": grouped.size(),
            "occupied": grouped.sum(),
            "occupancy_rate": grouped.mean(),
            "is_mixed": grouped.nunique().eq(2),
        }
    )
    return profile


def autocorrelation_report(
    df: pd.DataFrame, column: str = "temperature", lags: tuple[int, ...] = (1, 12, 288)
) -> dict[int, float]:
    """Lag autocorrelation of a column, with lags chosen to mean something.

    At a 5-minute interval: lag 1 = 5 minutes, lag 12 = 1 hour, lag 288 = 1 day.
    """
    series = df[column].astype(float)
    return {lag: float(series.autocorr(lag=lag)) for lag in lags}


# --------------------------------------------------------------------------
# Leakage checks
# --------------------------------------------------------------------------
def leakage_checks(
    train: pd.DataFrame,
    test: pd.DataFrame,
    tasks: tuple[str, ...] = ("occupancy", "temperature", "anomaly"),
    source: pd.DataFrame | None = None,
) -> list[Check]:
    """Prove the split cannot leak. Each check is a hard PASS/FAIL.

    `source` is the full pre-split frame. Supplying it enables the
    "nothing was dropped" check; without it that check is omitted rather than
    reported as a vacuous PASS.
    """
    checks: list[Check] = []

    max_train = train["timestamp"].max()
    min_test = test["timestamp"].min()
    checks.append(
        Check(
            "test period starts after training ends",
            bool(max_train < min_test),
            f"max(train)={max_train} < min(test)={min_test}",
        )
    )

    overlap = len(set(train["timestamp"]) & set(test["timestamp"]))
    checks.append(
        Check("no timestamp in both sets", overlap == 0, f"{overlap} overlapping timestamps")
    )

    if source is not None:
        dropped = len(source) - (len(train) + len(test))
        checks.append(
            Check(
                "no rows dropped by the split",
                dropped == 0,
                f"{len(train)} + {len(test)} = {len(train) + len(test)} "
                f"vs {len(source)} source rows",
            )
        )

    # Exact-duplicate feature rows across the boundary are a *mild* leakage
    # risk, and the honest test is not "no duplicates anywhere" but "no
    # duplicates that carry a contradictory answer".
    #
    # Phase 3 found 14 cross-boundary twin groups (15 of 576 test rows, 2.6%).
    # They arise naturally: in the small hours the room is dark, motion-free
    # and temperature drifts so slowly that it repeats to 2 decimals. All 14
    # share the same occupancy label, so a model that memorised them would
    # still be reproducing what the features already imply. Asserting zero
    # duplicates would therefore fail on harmless data while a genuine
    # contradiction - the case that actually corrupts a score - would be
    # measured just as loosely.
    is_twin, cross_groups, conflicting = cross_boundary_twin_mask(train, test)
    twin_test_rows = int(is_twin.sum())
    checks.append(
        Check(
            "no cross-boundary twins with contradictory labels",
            conflicting == 0,
            f"{cross_groups} twin groups, {twin_test_rows} test rows "
            f"({twin_test_rows / len(test) * 100:.1f}% of test), "
            f"{conflicting} contradictory",
        )
    )

    for task in tasks:
        features, target = TASKS[task]
        leaks = [c for c in features if c in (target,) or c in FORBIDDEN_INPUTS]
        checks.append(
            Check(
                f"no target leakage in '{task}' features",
                not leaks,
                f"features={features}, target={target}"
                if not leaks
                else f"LEAK: {leaks}",
            )
        )

    return checks


def describe_split(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, object]:
    """Summary numbers for the split, for the report."""
    return {
        "train_rows": len(train),
        "test_rows": len(test),
        "train_start": train["timestamp"].min(),
        "train_end": train["timestamp"].max(),
        "test_start": test["timestamp"].min(),
        "test_end": test["timestamp"].max(),
        "train_rate": float(train["is_anomaly"].mean()),
        "test_rate": float(test["is_anomaly"].mean()),
        "train_occupancy_rate": float(train["occupancy"].mean()),
        "test_occupancy_rate": float(test["occupancy"].mean()),
    }


if __name__ == "__main__":
    data = load_dataset()
    for c in validate_dataset(data):
        print(c.render())
    print()
    tr, te = chronological_split(data)
    for c in leakage_checks(tr, te, source=data):
        print(c.render())
