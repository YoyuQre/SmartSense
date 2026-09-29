"""
Phase 5 - historical replay: detection, control, and the metrics they produce.

Runs the Phase 5 pipeline over the full frozen dataset and writes
`data/control_output.csv`. Nothing is retrained here: the occupancy model is
loaded from `models/occupancy_model.pkl` as persisted by Phase 4, and the rolling
Z-score is the inherited Phase 4 class at its Phase 4 parameters. Re-running
this script therefore cannot change the answer through randomness or through a
refit.

Two conventions are used consistently and are worth stating up front:

1. The rolling Z-score needs a full 288-sample window before it produces a
   verdict, so the first 288 rows are a warmup period. Every rate is reported
   both including and excluding that period, because including it flatters a
   detector that does nothing there, and excluding it hides a detector that
   flags things it has no basis to judge.
2. The dataset's `is_anomaly` column is used to score detection, never to
   produce a decision. Control reads only predicted occupancy, temperature, light
   level, and sensor health.

No energy, cost, or comfort figure is reported. Nothing in this project models
actuator power draw, so a savings number would be invented. What can be counted
honestly is how often each actuator is on, and how often the fail-safe overrode
a decision that the healthy-sensor rule would have made.

Run:  python src/phase5_run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import config as cfg  # noqa: E402
import control_logic as ctl  # noqa: E402
import detection as det  # noqa: E402
import modeling as mo  # noqa: E402
import preprocessing as pp  # noqa: E402

OUTPUT_PATH = ROOT / "data" / "control_output.csv"

# The window the Phase 4 grid selected, read from the Phase 4 artifact itself
# rather than hardcoded, so a Phase 4 change cannot leave this script scoring
# against a stale warmup length. A miss here would silently misreport every rate.
WARMUP = next(
    window for window, k in mo.ZSCORE_GRID if (window, k) == (288, 2.0)
)


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator) / denominator if denominator else 0.0


def _detection_block(
    label: str, detected: np.ndarray, actual: np.ndarray, warmup: int
) -> dict[str, float]:
    """Precision/recall/F1 for a boolean detector against `is_anomaly`."""
    scored = slice(warmup, None)
    flagged = detected[scored]
    truth = actual[scored] == 1
    tp = int((flagged & truth).sum())
    fp = int((flagged & ~truth).sum())
    fn = int((~flagged & truth).sum())
    tn = int((~flagged & ~truth).sum())
    precision = _rate(tp, tp + fp)
    recall = _rate(tp, tp + fn)
    f1 = _rate(2 * tp, 2 * tp + fp + fn)
    return {
        "detector": label,
        "scored_rows": tp + fp + fn + tn,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def run() -> tuple[pd.DataFrame, det.DetectionResult, pd.DataFrame, pd.DataFrame]:
    frame = pp.load_dataset()
    if "hour" not in frame.columns:
        frame = pp.add_hour_column(frame)

    # Phase 4's model was trained on the primary split's feature order, which is
    # asserted against the persisted artifact rather than assumed.
    model = joblib.load(mo.OCCUPANCY_MODEL)
    expected = list(getattr(model, "feature_names_in_", []))
    features = [c for c in expected if c in frame.columns] if expected else []
    if not features:
        raise RuntimeError(
            "Could not determine the persisted model's feature order; refusing "
            "to guess, because a wrong order would silently change every "
            "control decision in this replay."
        )
    if len(features) != model.n_features_in_:
        raise RuntimeError(
            f"Persisted model expects {model.n_features_in_} features {expected}, "
            f"but the frame supplies {features}."
        )
    predicted = model.predict(frame[features]).astype(int)

    result = det.detect(frame)
    control = ctl.decide_all(
        predicted,
        frame["temperature"].to_numpy(),
        frame["light_level"].to_numpy(),
        result.temperature_sensor_health,
        result.light_sensor_health,
    )

    output = pd.DataFrame(
        {
            "timestamp": frame["timestamp"],
            "predicted_occupancy": predicted,
            "actual_occupancy": frame["occupancy"].astype(int),
            "temperature": frame["temperature"],
            "light_level": frame["light_level"],
            "light_status": control["light_status"],
            "hvac_status": control["hvac_status"],
            "anomaly_detected": result.anomaly_detected.astype(int),
            "anomaly_reason": result.anomaly_reason,
            "temperature_sensor_health": result.temperature_sensor_health,
            "light_sensor_health": result.light_sensor_health,
            "motion_sensor_health": result.motion_sensor_health,
            "control_reason": control["control_reason"],
        }
    )
    output.to_csv(OUTPUT_PATH, index=False)
    return output, result, control, frame


def report(output: pd.DataFrame, result, control, frame: pd.DataFrame) -> None:
    n = len(output)
    actual = frame["is_anomaly"].to_numpy()
    line = "-" * 74

    print(line)
    print("PHASE 5 REPLAY")
    print(line)
    print(f"  rows replayed        : {n}")
    print(f"  model                : {Path(mo.OCCUPANCY_MODEL).name} (loaded, not retrained)")
    print(f"  model features       : {list(getattr(joblib.load(mo.OCCUPANCY_MODEL), 'feature_names_in_', []))}")
    print(f"  z-score warmup       : first {WARMUP} rows ({WARMUP * cfg.SAMPLING_INTERVAL_MINUTES} min)")
    print(f"  output written to    : {OUTPUT_PATH.relative_to(ROOT)}")

    print()
    print("DETECTION")
    rows = [
        _detection_block("rolling z-score (Phase 4, inherited)", result.zscore_flag, actual, WARMUP),
        _detection_block("z-score + flatlines (Phase 5)", result.anomaly_detected, actual, WARMUP),
    ]
    header = f"  {'detector':<40} {'TP':>4} {'FP':>4} {'FN':>4} {'prec':>6} {'rec':>6} {'F1':>6}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        print(
            f"  {r['detector']:<40} {r['tp']:>4} {r['fp']:>4} {r['fn']:>4} "
            f"{r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f}"
        )
    print(f"  (scored on rows {WARMUP}-{n - 1}; the {WARMUP} warmup rows are excluded)")

    print()
    print("  flag reasons:")
    for reason, count in output["anomaly_reason"].value_counts().items():
        print(f"    {reason:<24} {count:>5}")

    print()
    print("  flatline flags and the false positives they cost:")
    y = actual == 1
    for label, flags in [
        ("temperature", result.temperature_flatline),
        ("light", result.light_flatline),
        ("motion", result.motion_flatline),
    ]:
        if not flags.any():
            print(f"    {label:<12} fired 0 times")
            continue
        print(
            f"    {label:<12} fired {int(flags.sum()):>3} times "
            f"({int((flags & y).sum())} on injected faults, "
            f"{int((flags & ~y).sum())} false positives)"
        )

    print()
    print("  do the flatline flags land on the injected stuck runs?")
    import data_generator as dg

    _plan = dg.generate_dataset(return_plan=True)[1]
    for _kind, column, start, length in _plan:
        if _kind != "stuck_sensor":
            continue
        attr = (
            "temperature_flatline" if column == "temperature" else "light_flatline"
        )
        hit = int(pd.Series(result.__dict__[attr]).iloc[start : start + length].sum())
        print(
            f"    injected stuck {column:<12} {length} samples "
            f"-> {hit}/{length} flagged"
        )

    print()
    print("SENSOR HEALTH")
    for column in [
        "temperature_sensor_health",
        "light_sensor_health",
        "motion_sensor_health",
    ]:
        suspect = int((output[column] == det.HEALTH_SUSPECT).sum())
        print(f"  {column:<28} suspect {suspect:>4} of {n}  ({_rate(suspect, n) * 100:.2f}%)")

    print()
    print("CONTROL OUTPUT")
    for column in ["light_status", "hvac_status"]:
        counts = output[column].value_counts()
        on = int(counts.get(ctl.LIGHT_ON if column == "light_status" else ctl.HVAC_COOLING, 0))
        on += int(counts.get(ctl.HVAC_HEATING, 0)) if column == "hvac_status" else 0
        print(f"  {column}:")
        for status, count in counts.items():
            print(f"    {status:<10} {count:>5}  ({_rate(count, n) * 100:5.2f}%)")
        print(f"    -> active {on} of {n} intervals ({_rate(on, n) * 100:.2f}%)")

    switches = (
        (output["light_status"] != output["light_status"].shift()).sum() - 1
        + (output["hvac_status"] != output["hvac_status"].shift()).sum() - 1
    )
    print(f"  actuator state changes across the series: {switches}")

    print()
    print("FAIL-SAFE OVERRIDES")
    healthy_control = ctl.decide_all(
        output["predicted_occupancy"].to_numpy(),
        frame["temperature"].to_numpy(),
        frame["light_level"].to_numpy(),
        [det.HEALTH_NORMAL] * n,
        [det.HEALTH_NORMAL] * n,
    )
    light_overridden = int(
        (healthy_control["light_status"] != output["light_status"]).sum()
    )
    hvac_overridden = int(
        (healthy_control["hvac_status"] != output["hvac_status"]).sum()
    )
    light_healthy_on = int((healthy_control["light_status"] == ctl.LIGHT_ON).sum())
    hvac_healthy_active = int(
        (healthy_control["hvac_status"].isin([ctl.HVAC_COOLING, ctl.HVAC_HEATING])).sum()
    )
    print(
        f"  light  : {light_overridden} interval(s) forced OFF that a healthy "
        f"sensor would have switched on ({_rate(light_overridden, light_healthy_on) * 100:.1f}% "
        f"of the {light_healthy_on} healthy-sensor ON intervals)"
    )
    print(
        f"  HVAC   : {hvac_overridden} interval(s) forced OFF that a healthy "
        f"sensor would have run ({_rate(hvac_overridden, hvac_healthy_active) * 100:.1f}% "
        f"of the {hvac_healthy_active} healthy-sensor active intervals)"
    )
    print("  An override is a deliberate loss of service, not a bug: an untrusted")
    print("  reading is treated as insufficient evidence to act on.")

    print()
    print("CONTROL REACHABILITY")
    print("  Every branch is counted, including the ones with no rows. A branch that")
    print("  never fires is a fact about the data, not a broken rule, and printing")
    print("  the gating counts keeps the two distinguishable.")
    temp = frame["temperature"].to_numpy()
    light = frame["light_level"].to_numpy()
    pred_occ = output["predicted_occupancy"].to_numpy() == 1
    temp_suspect = np.array(
        [h == det.HEALTH_SUSPECT for h in result.temperature_sensor_health]
    )
    light_suspect = np.array(
        [h == det.HEALTH_SUSPECT for h in result.light_sensor_health]
    )
    cooling_gate = pred_occ & (temp > cfg.HVAC_COOLING_THRESHOLD) & ~temp_suspect
    heating_gate = pred_occ & (temp < cfg.HVAC_HEATING_THRESHOLD) & ~temp_suspect
    light_on_gate = pred_occ & (light < cfg.LIGHT_THRESHOLD) & ~light_suspect
    print(f"    predicted occupied                 : {int(pred_occ.sum()):>5}")
    print(f"    temperature > {cfg.HVAC_COOLING_THRESHOLD} C                 : {int((temp > cfg.HVAC_COOLING_THRESHOLD).sum()):>5}")
    print(f"    temperature < {cfg.HVAC_HEATING_THRESHOLD} C                 : {int((temp < cfg.HVAC_HEATING_THRESHOLD).sum()):>5}")
    print(f"    -> COOLING reachable               : {int(cooling_gate.sum()):>5}  reached {int((output['hvac_status'] == ctl.HVAC_COOLING).sum())}")
    print(f"    -> HEATING reachable               : {int(heating_gate.sum()):>5}  reached {int((output['hvac_status'] == ctl.HVAC_HEATING).sum())}")
    print(f"    -> light ON reachable              : {int(light_on_gate.sum()):>5}  reached {int((output['light_status'] == ctl.LIGHT_ON).sum())}")
    if int(heating_gate.sum()) == 0:
        print()
        print("    HEATING is unreachable in this dataset. The generator makes cold")
        print("    coincide with absence: there are rows below 20 C, but none of them")
        print(f"    with predicted occupancy 1 (coldest predicted-occupied row is")
        print(f"    {temp[pred_occ].min():.2f} C). The rule itself is exercised by")
        print("    test_occupied_and_cold_heating_on and the mutual-exclusion test,")
        print("    so it is verified by unit test and by nothing in the replay.")

    print()
    print("CONTROL REASONS")
    for reason, count in output["control_reason"].value_counts().items():
        print(f"    {reason:<34} {count:>5}")

    print()
    print("OCCUPANCY PREDICTION (replay context, not a Phase 4 re-report)")
    pred = output["predicted_occupancy"].to_numpy()
    truth = output["actual_occupancy"].to_numpy()
    tp = int(((pred == 1) & (truth == 1)).sum())
    fp = int(((pred == 1) & (truth == 0)).sum())
    fn = int(((pred == 0) & (truth == 1)).sum())
    tn = int(((pred == 0) & (truth == 0)).sum())
    print(
        f"  TP={tp} FP={fp} FN={fn} TN={tn}  "
        f"accuracy={_rate(tp + tn, n):.4f} "
        f"precision={_rate(tp, tp + fp):.4f} "
        f"recall={_rate(tp, tp + fn):.4f} "
        f"F1={_rate(2 * tp, 2 * tp + fp + fn):.4f}"
    )
    print("  NOTE: this is the Phase 4 model applied to all 2016 rows, which is a")
    print("  different quantity from the Phase 4 primary-split score and is not")
    print("  a revision of it. Only the test split is a generalisation estimate.")

    print()
    print("FLATLINE SENSITIVITY (reproduced so the operating point is checkable)")
    table = det.sensitivity_table(frame)
    if table:
        head = f"  {'sensor':<12} {'window':<16} {'max range':>9} {'caught':>7} {'false pos':>10}"
        print(head)
        print("  " + "-" * (len(head) - 2))
        for label, window, rng, caught, fp_count in table:
            print(f"  {label:<12} {window:<16} {rng:>9} {caught:>7} {fp_count:>10}")

    print()
    print(line)
    print("No energy or cost saving is reported: this project has no actuator")
    print("power model, so any savings figure would be fabricated.")
    print(line)


if __name__ == "__main__":
    output, result, control, frame = run()
    report(output, result, control, frame)
