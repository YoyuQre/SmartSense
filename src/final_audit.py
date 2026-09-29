"""
Phase 7 - final integration audit.

Re-checks every claim the project makes, in the two groups the brief calls for:
FUNCTIONAL (does each feature actually work) and REGRESSION (has anything
drifted since the phase that produced it).

This exists as a runnable script rather than a one-off command because the
project's credibility rests on the claim that nothing was faked and nothing
drifted. A claim that can only be checked by re-reading the source is a claim
that quietly stops being true. Run it before the viva and again after any edit:

    python src/final_audit.py

Exit code 0 means every check passed. A non-zero exit means at least one
regression, which should be fixed rather than explained away.
"""

from __future__ import annotations

import ast
import hashlib
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import config as cfg  # noqa: E402
import dashboard_data as dd  # noqa: E402

RULE = "=" * 74
THIN = "-" * 74

# Recorded when each phase froze its artifact. A mismatch here is the single
# most important thing this script can catch.
FROZEN = {
    "data/sensor_data.csv": (
        "2D38D7416FDD8D80FE78C52E383BCDC50943722CEB346AD9B2B46ED5BED7D091",
        "Phase 2C",
    ),
    "models/occupancy_model.pkl": (
        "86E55F6A18F3117453CC585962E2013460756B0FC4CBA46CF6EFC79BAEE68CF1",
        "Phase 4",
    ),
    "models/temperature_model.pkl": (
        "83024A3C57F1C3C3E1643DE1F8D57A9445D9353F4966FBBB0F6E2329424DBBB1",
        "Phase 4",
    ),
    "src/modeling.py": (
        "BC32D391D7BDFF304B339305C0C993F656717C3A0C554F72C9ACF7E597B707D2",
        "Phase 4",
    ),
    "src/preprocessing.py": (
        "29881C90EFA11E9D51D700702290FE2E10E6425B4CADD081CFE1947CC9652584",
        "Phase 4",
    ),
    # Phase 5 replay output. Pinned so a re-run that silently changes a single
    # decision is a failure, not a surprise. src/phase5_run.py is deterministic
    # and tests assert a re-run is byte-identical, so this hash is stable.
    "data/control_output.csv": (
        "EAA070CEFEEF48C2ED68344B84A40BF46E8BC70B29BB2FD655C33ADAB23D136D",
        "Phase 5",
    ),
}

TRAINING_APIS = {"fit", "fit_transform", "partial_fit", "train_test_split", "GridSearchCV", "cross_val_score"}
FIT_HELPERS = ["fit_occupancy", "fit_temperature", "tune_decision_tree"]

PHASE4_PRIMARY_EXPECTED = {
    "accuracy": 0.7778, "precision": 0.5451, "recall": 0.9539, "f1": 0.6938,
    "tn": 303, "fp": 121, "fn": 7, "tp": 145,
    "mae": 1.3404, "rmse": 1.9532, "r2": 0.6715,
}
PHASE5_EXPECTED = {
    "rows": 2016, "injected": 40,
    "tp": 25, "fp": 10, "fn": 8, "f1": 0.735,
    "temperature_flatline": 12, "light_flatline": 8, "motion_flatline": 0,
}


class Audit:
    def __init__(self) -> None:
        self.passed: list[str] = []
        self.failed: list[tuple[str, str]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        (self.passed if ok else self.failed).append((name, detail))
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}" + (f"  -  {detail}" if detail else ""))
        return ok

    def section(self, title: str) -> None:
        print()
        print(title)
        print(THIN)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def called_attributes(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                out.add(node.func.attr)
            elif isinstance(node.func, ast.Name):
                out.add(node.func.id)
    return out


def run(command: list[str], cwd: Path) -> tuple[int, str]:
    try:
        done = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=900)
        return done.returncode, (done.stdout + done.stderr)
    except Exception as exc:  # noqa: BLE001
        return 1, f"{type(exc).__name__}: {exc}"


def main() -> int:
    a = Audit()
    print(RULE)
    print("PHASE 7 - FINAL INTEGRATION AUDIT")
    print(RULE)

    # ---------------------------------------------------------------- FUNCTIONAL
    a.section("FUNCTIONAL")

    for doc in ["app.py", "README.md", "docs/PROJECT_REPORT.md", "docs/VIVA.md",
                "docs/DEMO_PROCEDURE.md", "docs/LIMITATIONS.md",
                "docs/ARCHITECTURE.md"]:
        path = ROOT / doc
        a.check(f"{doc} exists", path.exists(), f"{path.stat().st_size} bytes" if path.exists() else "missing")

    diagram = ROOT / "docs" / "architecture.png"
    a.check("architecture diagram exists", diagram.exists(),
            f"{diagram.stat().st_size} bytes" if diagram.exists() else "missing")

    app_source = (ROOT / "app.py").read_text(encoding="utf-8")
    a.check("dashboard declares Simulation Mode", "SIMULATION_BANNER" in app_source)
    a.check("dashboard gates serial mode on hardware", "No ESP32 is connected" in app_source)
    a.check("dashboard never claims energy savings",
            "No energy or cost saving is reported" in app_source)

    dataset = dd.load_dataset()
    a.check("dataset loads", len(dataset) == 2016, f"{len(dataset)} rows, {len(dataset.columns)} columns")

    replay = dd.load_replay()
    required = [
        "timestamp", "predicted_occupancy", "actual_occupancy", "temperature",
        "light_level", "light_status", "hvac_status", "anomaly_detected",
        "anomaly_reason", "temperature_sensor_health", "light_sensor_health",
        "motion_sensor_health", "control_reason",
    ]
    missing = [c for c in required if c not in replay.columns]
    a.check("control output has all required columns", not missing, f"missing: {missing}" if missing else "13 columns")
    a.check("control output has no missing values", not replay.isna().any().any())

    counts = dd.phase5_counts()
    a.check("filters return rows", len(dd.filter_replay(replay, recent_n=100)) == 100, "recent_n=100")
    a.check("anomaly-only filter works",
            (dd.filter_replay(replay, anomaly_only=True, recent_n=None)["anomaly_detected"] == 1).all())
    a.check("sensor-suspect filter works",
            len(dd.filter_replay(replay, suspect_only=True, recent_n=None)) > 0)
    a.check("control table is capped below full length",
            dd.MAX_TABLE_ROWS < len(replay), f"cap {dd.MAX_TABLE_ROWS} of {len(replay)}")

    metrics = dd.phase4_metrics()
    a.check("dashboard recomputes Phase 4 metrics", bool(metrics), "inference only, no retraining")

    candidates, rejected = dd.scan_serial_ports()
    a.check("serial mode offers no device without hardware", candidates == [],
            f"{len(rejected)} port(s) correctly rejected as non-ESP32")

    # --------------------------------------------------------------- REGRESSION
    a.section("REGRESSION")

    for relative, (expected, phase) in FROZEN.items():
        path = ROOT / relative
        actual = sha256(path) if path.exists() else "MISSING"
        a.check(f"{relative} unchanged since {phase}", actual == expected,
                actual[:16] + "..." if actual == expected else f"expected {expected[:16]}... got {actual[:16]}...")

    for source in ["app.py", "src/dashboard_data.py", "src/detection.py",
                   "src/control_logic.py", "src/phase5_run.py"]:
        path = ROOT / source
        offenders = called_attributes(path) & TRAINING_APIS
        a.check(f"{source} performs no training", not offenders,
                f"calls {sorted(offenders)}" if offenders else "no training APIs")

    for source in ["src/dashboard_data.py", "app.py"]:
        text = (ROOT / source).read_text(encoding="utf-8")
        leaked = [f for f in FIT_HELPERS if f in text]
        a.check(f"{source} does not call Phase 4 fit helpers", not leaked, f"uses {leaked}" if leaked else "")

    try:
        occupancy = joblib.load(ROOT / "models" / "occupancy_model.pkl")
        temperature = joblib.load(ROOT / "models" / "temperature_model.pkl")
        a.check("persisted models load", True,
                f"occupancy {type(occupancy).__name__}, temperature {type(temperature).__name__}")
    except Exception as exc:  # noqa: BLE001
        a.check("persisted models load", False, str(exc))

    primary = metrics.get("primary", {})
    metric_ok = all(
        abs(float(primary.get(k, -1)) - v) <= 5e-4
        for k, v in PHASE4_PRIMARY_EXPECTED.items()
    )
    a.check("dashboard metrics match Phase 4 report", metric_ok,
            f"acc {primary.get('accuracy', 0):.4f}, F1 {primary.get('f1', 0):.4f}, "
            f"MAE {primary.get('mae', 0):.4f}")

    phase5_ok = (
        counts.total_rows == PHASE5_EXPECTED["rows"]
        and counts.injected_anomalies == PHASE5_EXPECTED["injected"]
        and counts.true_positives == PHASE5_EXPECTED["tp"]
        and counts.false_positives == PHASE5_EXPECTED["fp"]
        and counts.false_negatives == PHASE5_EXPECTED["fn"]
        and abs(counts.f1 - PHASE5_EXPECTED["f1"]) <= 5e-3
        and counts.temperature_flatline == PHASE5_EXPECTED["temperature_flatline"]
        and counts.light_flatline == PHASE5_EXPECTED["light_flatline"]
        and counts.motion_flatline == PHASE5_EXPECTED["motion_flatline"]
    )
    a.check("anomaly counts match Phase 5", phase5_ok,
            f"TP {counts.true_positives} FP {counts.false_positives} FN {counts.false_negatives} "
            f"F1 {counts.f1:.3f}")

    code, output = run([sys.executable, "-m", "pytest", "tests", "-q"], ROOT)
    tail = [line for line in output.splitlines() if "passed" in line or "failed" in line]
    a.check("pytest suite green", code == 0, tail[-1].strip() if tail else f"exit {code}")

    code, output = run([sys.executable, "-m", "platformio", "run"], ROOT)
    ram = next((l.strip() for l in output.splitlines() if "RAM:" in l), "")
    flash = next((l.strip() for l in output.splitlines() if "Flash:" in l), "")
    a.check("firmware builds", code == 0 and "SUCCESS" in output, f"{ram} | {flash}")

    main_cpp = ROOT / "src" / "main.cpp"
    sketch = ROOT / "sketch.ino"
    identical = sha256(main_cpp) == sha256(sketch)
    a.check("firmware copies byte-identical", identical, "src/main.cpp == sketch.ino")

    summary = dd.control_summary(replay)
    a.check("fail-safe invariant holds across all rows", summary["fail_safe_violations"] == 0,
            "no suspect sensor drove an actuator")

    # ----------------------------------------------------------------- WORDING
    a.section("WORDING (claims that must not overstate the work)")

    banned = ["energy-saving system", "energy saving system", "proven energy savings",
              "reduced energy consumption by", "achieves energy"]
    # Narrative documents only. docs/APPENDIX_CODE.md is a generated verbatim dump
    # of every source file, so it necessarily contains this script's own banned-phrase
    # list and any code comment - neither of which is a claim about the project.
    # The exclusion is printed so it is auditable rather than silent.
    narrative = [
        ROOT / "README.md",
        ROOT / "docs" / "PROJECT_REPORT.md",
        ROOT / "docs" / "ARCHITECTURE.md",
        ROOT / "docs" / "VIVA.md",
        ROOT / "docs" / "DEMO_PROCEDURE.md",
        ROOT / "docs" / "LIMITATIONS.md",
    ]
    excluded = [ROOT / "docs" / "APPENDIX_CODE.md"]
    scanned = [d for d in narrative if d.exists()]
    offenders: list[str] = []
    for doc in scanned:
        text = doc.read_text(encoding="utf-8").lower()
        for phrase in banned:
            if phrase in text:
                offenders.append(f"{doc.name}: '{phrase}'")
    a.check(f"no narrative document claims energy savings ({len(scanned)} scanned, "
            f"{len([e for e in excluded if e.exists()])} code listing excluded)",
            not offenders,
            "; ".join(offenders) if offenders else "wording is defensible")

    for doc in ["README.md", "docs/PROJECT_REPORT.md", "docs/LIMITATIONS.md"]:
        path = ROOT / doc
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8").lower()
        simulated = "simulat" in text
        honest = "not" in text or "no " in text
        a.check(f"{doc} discloses the simulated dataset", simulated and honest)

    # ------------------------------------------------------------------ REPORT
    print()
    print(RULE)
    total = len(a.passed) + len(a.failed)
    print(f"RESULT: {len(a.passed)}/{total} checks passed")
    if a.failed:
        print()
        print("FAILURES:")
        for name, detail in a.failed:
            print(f"  - {name}" + (f": {detail}" if detail else ""))
        print(RULE)
        return 1
    print("All functional and regression checks passed.")
    print(RULE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
