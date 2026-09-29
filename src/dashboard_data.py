"""
Phase 6 - data access layer for the dashboard.

This module is deliberately free of any Streamlit import. All caching is applied
in `app.py` as thin wrappers, which keeps every function here callable from a
plain pytest process. That matters because the acceptance criteria for Phase 6
("metrics match Phase 4", "anomaly counts match Phase 5", "source dataset
SHA-256 unchanged") are only worth anything if they are actually asserted by
tests, and a Streamlit-coupled module is awkward to test.

Three rules this module exists to enforce:

1. **No retraining, ever.** Models are opened with `joblib.load` and used for
   inference only. Nothing here calls `.fit`, and a test asserts that.
2. **The frozen inputs are verified, not assumed.** The dataset and both model
   files are hashed on load and compared against the recorded Phase 2/4 hashes.
   A dashboard that silently reads a modified dataset is worse than no
   dashboard, so the hash is displayed and a mismatch is surfaced in the UI.
3. **Numbers carry their provenance.** Phase 4 primary metrics and the weekday
   diagnostic are different quantities produced by different models, and this
   module keeps them apart and labels them, because conflating them is the
   easiest way to end up quoting a number that means something else.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import config as cfg  # noqa: E402
import detection as det  # noqa: E402
import modeling as mo  # noqa: E402
import preprocessing as pp  # noqa: E402

DATASET_PATH = ROOT / "data" / "sensor_data.csv"
REPLAY_PATH = ROOT / "data" / "control_output.csv"
OCCUPANCY_MODEL = mo.OCCUPANCY_MODEL
TEMPERATURE_MODEL = mo.TEMPERATURE_MODEL

# Recorded in Phases 2C/4 and asserted on every dashboard load.
FROZEN_SHA256 = {
    "data/sensor_data.csv":
        "2D38D7416FDD8D80FE78C52E383BCDC50943722CEB346AD9B2B46ED5BED7D091",
    "models/occupancy_model.pkl":
        "86E55F6A18F3117453CC585962E2013460756B0FC4CBA46CF6EFC79BAEE68CF1",
    "models/temperature_model.pkl":
        "83024A3C57F1C3C3E1643DE1F8D57A9445D9353F4966FBBB0F6E2329424DBBB1",
}

# Phase 4 as reported in README section 12. The dashboard recomputes these from
# the persisted models and a test asserts the recomputed values equal these, so
# the acceptance criterion "metrics match Phase 4" is checked rather than
# asserted in a comment.
PHASE4_PRIMARY = {
    "accuracy": 0.7778, "precision": 0.5451, "recall": 0.9539, "f1": 0.6938,
    "tn": 303, "fp": 121, "fn": 7, "tp": 145, "n_test": 576,
    "mae": 1.3404, "rmse": 1.9532, "r2": 0.6715,
}
PHASE4_WEEKDAY_SEPARATE_MODEL = {
    "accuracy": 0.8472, "precision": 0.9044, "recall": 0.7987, "f1": 0.8483,
    "mae": 1.3165, "rmse": 1.6541, "r2": 0.7606, "n_test": 288,
}

WEEKDAY_CAVEAT = (
    "This row is the persisted PRIMARY model applied to the Wednesday holdout. "
    "The Phase 4 figure of F1 0.8483 came from a model trained separately on the "
    "weekday training split, which was never persisted, so the two numbers are "
    "not the same quantity. The primary model is shown here deliberately: it "
    "lets distribution shift be separated from model weakness using one fixed "
    "model."
)

SIMULATION_BANNER = "Simulation Mode - historical sensor data"
SIMULATION_DETAIL = (
    "Every value on this page comes from the frozen files data/sensor_data.csv "
    "and data/control_output.csv. No ESP32 is streaming data, and nothing here "
    "is live."
)

# Mode labels are defined here and imported by app.py so the tests can assert on
# the same strings the UI renders, instead of duplicating literals that could
# drift apart.
MODE_SIMULATION_LABEL = "Historical Simulation (frozen data)"
MODE_SERIAL_LABEL = "ESP32 Serial (live hardware)"

MAX_TABLE_ROWS = 500


# --------------------------------------------------------------------------
# Integrity
# --------------------------------------------------------------------------
def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def verify_frozen_files() -> list[dict[str, str]]:
    """Hash every frozen input and flag any drift.

    Displayed in the UI so a reader can see that the dashboard is reading the
    same bytes the earlier phases produced.
    """
    rows: list[dict[str, str]] = []
    for relative, expected in FROZEN_SHA256.items():
        path = ROOT / relative
        if not path.exists():
            rows.append({"file": relative, "status": "MISSING", "expected": expected, "actual": "-"})
            continue
        actual = sha256_of(path)
        rows.append({
            "file": relative,
            "status": "unchanged" if actual == expected else "CHANGED",
            "expected": expected[:16] + "...",
            "actual": actual[:16] + "...",
        })
    return rows


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_dataset() -> pd.DataFrame:
    frame = pp.load_dataset(DATASET_PATH)
    if "hour" not in frame.columns:
        frame = pp.add_hour_column(frame)
    return frame


def load_replay() -> pd.DataFrame:
    """Load the Phase 5 control output. Never recomputed here.

    The dashboard displays what Phase 5 actually produced rather than
    re-deriving it, so a Phase 5 regression cannot be hidden by the dashboard
    quietly recomputing a different answer.
    """
    frame = pd.read_csv(REPLAY_PATH)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame


def load_models() -> tuple[object, object, list[str]]:
    """Load the persisted Phase 4 models. Inference only, never fitted."""
    occupancy = joblib.load(OCCUPANCY_MODEL)
    temperature = joblib.load(TEMPERATURE_MODEL)
    features = list(getattr(occupancy, "feature_names_in_", []))
    return occupancy, temperature, features


# --------------------------------------------------------------------------
# Phase 4 metrics, recomputed by inference on the frozen splits
# --------------------------------------------------------------------------
def phase4_metrics() -> dict[str, dict[str, float]]:
    """Recompute Phase 4 metrics from the persisted models.

    Only the primary models were persisted, so the primary row is exactly
    reproducible. The weekday row is the same primary model scored on the
    weekday holdout, which is a different quantity from Phase 4's separately
    trained weekday model and is labelled as such in the UI.
    """
    frame = load_dataset()
    occupancy, temperature, features = load_models()
    _primary_train, primary_test = pp.chronological_split(frame)
    _holdout_train, holdout_test = pp.weekday_holdout_split(frame)

    out: dict[str, dict[str, float]] = {}
    for label, test in [("primary", primary_test), ("weekday", holdout_test)]:
        y_true, y_pred = mo.occupancy_scores(occupancy, test, features)
        metrics = dict(mo.classification_metrics(y_true, y_pred))
        t_true, t_pred = mo.temperature_scores(temperature, test)
        metrics.update(mo.regression_metrics(t_true, t_pred))
        metrics["n_test"] = int(len(test))
        out[label] = metrics
    return out


def phase4_agreement() -> dict[str, bool]:
    """Whether recomputed primary metrics equal the recorded Phase 4 report."""
    metrics = phase4_metrics()["primary"]
    keys = ["accuracy", "precision", "recall", "f1", "tn", "fp", "fn", "tp", "n_test"]
    temperature_keys = ["mae", "rmse", "r2"]
    return {
        "occupancy": all(
            _close(metrics[k], PHASE4_PRIMARY[k]) for k in keys
        ),
        "temperature": all(
            _close(metrics[k], PHASE4_PRIMARY[k]) for k in temperature_keys
        ),
    }


def _close(a: float, b: float, tol: float = 5e-4) -> bool:
    return abs(float(a) - float(b)) <= tol


# --------------------------------------------------------------------------
# Phase 5 counts, read from the frozen output
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DetectionCounts:
    total_rows: int
    injected_anomalies: int
    flagged: int
    zscore_only: int
    flatline_total: int
    temperature_flatline: int
    light_flatline: int
    motion_flatline: int
    warmup_rows: int
    scorable_anomalies: int
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int

    @property
    def precision(self) -> float:
        return self.true_positives / (self.true_positives + self.false_positives) if (self.true_positives + self.false_positives) else 0.0

    @property
    def recall(self) -> float:
        return self.true_positives / (self.true_positives + self.false_negatives) if (self.true_positives + self.false_negatives) else 0.0

    @property
    def f1(self) -> float:
        denominator = 2 * self.true_positives + self.false_positives + self.false_negatives
        return 2 * self.true_positives / denominator if denominator else 0.0


WARMUP_ROWS = 288  # the inherited rolling Z-score cannot judge its first window


def phase4_baselines() -> dict[str, dict[str, float]]:
    """Trivial baselines computed on each frozen test split.

    Computed rather than hardcoded. A bare accuracy figure is hard to judge, and
    a hardcoded baseline is a trap: the always-unoccupied accuracy depends on
    the occupancy rate *of the test split*, which is 0.5236 on the primary
    training window but only 0.2639 on the primary test window. Using the wrong
    one turns a genuine +0.042 improvement into a fictitious +0.3014.
    """
    frame = load_dataset()
    _primary_train, primary_test = pp.chronological_split(frame)
    _holdout_train, holdout_test = pp.weekday_holdout_split(frame)

    out: dict[str, dict[str, float]] = {}
    for label, test in [("primary", primary_test), ("weekday", holdout_test)]:
        y = test["occupancy"].to_numpy()
        out[label] = {
            "occupancy_rate": float(y.mean()),
            "always_unoccupied": float((y == 0).mean()),
            "always_occupied": float((y == 1).mean()),
        }
    return out


def phase5_counts() -> DetectionCounts:
    """Anomaly counts derived from the frozen Phase 5 output.

    Scored on rows 288+ for the same reason Phase 5 does: the inherited
    detector has no baseline for its first window, so including those rows
    would credit it for silence.
    """
    frame = load_dataset()
    replay = load_replay()
    detected = replay["anomaly_detected"].to_numpy() == 1
    actual = frame["is_anomaly"].to_numpy() == 1

    scored = slice(WARMUP_ROWS, None)
    flagged = detected[scored]
    truth = actual[scored]
    tp = int((flagged & truth).sum())
    fp = int((flagged & ~truth).sum())
    fn = int((~flagged & truth).sum())
    tn = int((~flagged & ~truth).sum())

    reason = replay["anomaly_reason"]
    flatline = reason.isin([
        det.REASON_TEMP_FLAT, det.REASON_LIGHT_FLAT, det.REASON_MOTION_FLAT
    ])

    return DetectionCounts(
        total_rows=len(replay),
        injected_anomalies=int(actual.sum()),
        flagged=int(detected.sum()),
        zscore_only=int((reason == det.REASON_ZSCORE).sum()),
        flatline_total=int(flatline.sum()),
        temperature_flatline=int((reason == det.REASON_TEMP_FLAT).sum()),
        light_flatline=int((reason == det.REASON_LIGHT_FLAT).sum()),
        motion_flatline=int((reason == det.REASON_MOTION_FLAT).sum()),
        warmup_rows=WARMUP_ROWS,
        scorable_anomalies=int(truth.sum()),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
    )


# --------------------------------------------------------------------------
# Current / historical state
# --------------------------------------------------------------------------
def latest_observation(replay: pd.DataFrame) -> dict[str, object]:
    """The most recent row of the historical replay.

    In simulation mode this is a historical record, not a live reading, and the
    UI labels it with its timestamp for exactly that reason.
    """
    row = replay.iloc[-1]
    healthy = (
        row["temperature_sensor_health"] == det.HEALTH_NORMAL
        and row["light_sensor_health"] == det.HEALTH_NORMAL
        and row["motion_sensor_health"] == det.HEALTH_NORMAL
    )
    return {
        "timestamp": row["timestamp"],
        "predicted_occupancy": int(row["predicted_occupancy"]),
        "temperature": float(row["temperature"]),
        "light_level": int(row["light_level"]),
        "light_status": row["light_status"],
        "hvac_status": row["hvac_status"],
        "anomaly_detected": bool(row["anomaly_detected"]),
        "anomaly_reason": row["anomaly_reason"],
        "sensor_health": "Normal" if healthy else "Suspect",
        "control_reason": row["control_reason"],
    }


def control_summary(replay: pd.DataFrame) -> dict[str, object]:
    n = len(replay)
    light_on = int((replay["light_status"] == "ON").sum())
    cooling = int((replay["hvac_status"] == "COOLING").sum())
    heating = int((replay["hvac_status"] == "HEATING").sum())
    active = cooling + heating
    overrides = int(
        (
            (replay["temperature_sensor_health"] == det.HEALTH_SUSPECT)
            & (replay["hvac_status"] != "OFF")
        ).sum()
    ) + int(
        (
            (replay["light_sensor_health"] == det.HEALTH_SUSPECT)
            & (replay["light_status"] != "OFF")
        ).sum()
    )
    return {
        "rows": n,
        "light_on": light_on,
        "light_on_pct": 100.0 * light_on / n,
        "cooling": cooling,
        "heating": heating,
        "hvac_active": active,
        "hvac_active_pct": 100.0 * active / n,
        "fail_safe_violations": overrides,
    }


# --------------------------------------------------------------------------
# Filtering for the control table
# --------------------------------------------------------------------------
def filter_replay(
    replay: pd.DataFrame,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    recent_n: int | None = 100,
    anomaly_only: bool = False,
    suspect_only: bool = False,
) -> pd.DataFrame:
    """Apply the dashboard's filters.

    `recent_n` is applied last so that "recent N" means recent within the
    filtered selection rather than recent within the whole file, which is the
    behaviour a reader expects from a filter panel.
    """
    frame = replay.copy()
    if start is not None:
        frame = frame[frame["timestamp"] >= pd.Timestamp(start)]
    if end is not None:
        frame = frame[frame["timestamp"] <= pd.Timestamp(end)]
    if anomaly_only:
        frame = frame[frame["anomaly_detected"] == 1]
    if suspect_only:
        frame = frame[
            (frame["temperature_sensor_health"] == det.HEALTH_SUSPECT)
            | (frame["light_sensor_health"] == det.HEALTH_SUSPECT)
            | (frame["motion_sensor_health"] == det.HEALTH_SUSPECT)
        ]
    if recent_n is not None and len(frame) > recent_n:
        frame = frame.tail(recent_n)
    return frame.reset_index(drop=True)


# --------------------------------------------------------------------------
# ESP32 serial (real hardware only)
# --------------------------------------------------------------------------
# Descriptions and USB IDs that indicate a plausible ESP32 dev board. Anything
# else - notably a Bluetooth SPP COM port, which is what this machine actually
# has - is rejected, because offering to "connect" to one and then showing
# nothing would be indistinguishable from fabricating data.
ESP32_HINTS = ("esp32", "esp-32", "espressif", "ch340", "cp210", "silabs", "silicon labs", "usb jtag", "wch")
ESP32_VENDOR_IDS = {0x303A, 0x1A86, 0x10C4, 0x1B4F, 0x0403}


def scan_serial_ports() -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    """Split available COM ports into plausible-ESP32 and rejected.

    Returns `(candidates, rejected)`. A port is only offered for connection if
    it advertises an ESP32-ish description or a known USB-serial vendor ID;
    otherwise it is listed with the reason it was rejected.
    """
    try:
        from serial.tools import list_ports
    except ImportError:
        return [], [{"device": "-", "description": "pyserial is not installed", "reason": "missing dependency"}]

    candidates: list[dict[str, object]] = []
    rejected: list[dict[str, str]] = []
    for port in list_ports.comports():
        description = (port.description or "").lower()
        vendor = (port.vid if port.vid else 0)
        matches = (vendor in ESP32_VENDOR_IDS) or any(h in description for h in ESP32_HINTS)
        entry = {
            "device": port.device,
            "description": port.description,
            "vid": None if port.vid is None else f"0x{port.vid:04X}",
            "pid": None if port.pid is None else f"0x{port.pid:04X}",
        }
        if matches:
            candidates.append(entry)
        else:
            rejected.append({
                "device": str(port.device),
                "description": str(port.description),
                "reason": "not an ESP32 USB-serial device (no matching USB vendor ID or description)",
            })
    return candidates, rejected


def parse_serial_lines(text: str) -> tuple[list[dict[str, float]], list[str]]:
    """Parse the firmware's serial protocol from README section 8.

    Data records look like:
        DATA,5000,0,742,24.31,0,0
    Error records look like:
        ERR,TEMP_INVALID,raw=-127.00,action=HVAC_FORCED_OFF

    The startup banner is free-form text rather than a fixed set of prefixes, so
    it is identified positionally: anything before the first DATA/ERR line is
    banner. Once the stream has started, an unrecognised line is reported as a
    problem instead of being swallowed, so a protocol change surfaces as a
    visible error rather than a quietly empty chart.
    """
    records: list[dict[str, float]] = []
    errors: list[str] = []
    started = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("ERR,"):
            started = True
            errors.append(line)
            continue
        if line.startswith("DATA,"):
            started = True
            parts = line.split(",")
            if len(parts) != 7:
                errors.append(f"malformed DATA record: {line[:60]}")
                continue
            try:
                records.append({
                    "timestamp_ms": float(parts[1]),
                    "motion": float(parts[2]),
                    "light_level": float(parts[3]),
                    "temperature": float(parts[4]),
                    "light_status": float(parts[5]),
                    "hvac_status": float(parts[6]),
                })
            except ValueError:
                errors.append(f"non-numeric field in: {line[:60]}")
            continue
        if not started:
            continue  # startup banner
        errors.append(f"unrecognised line: {line[:60]}")
    return records, errors
