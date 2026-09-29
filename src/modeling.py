"""
Phase 4 - model training, evaluation and persistence.

Run:  python src/modeling.py

Trains the three Phase 4 components and evaluates each on TWO splits:

  primary   chronological, 2026-01-05..09 train -> 01-10..11 test
            (Mon-Fri train, Sat-Sun test; the deployment-realistic case)

  weekday   diagnostic only: hold out Wednesday, train on the other six days
            including both weekends. Not chronologically pure - training
            contains days after the holdout - which is exactly why it is a
            diagnostic and the chronological split stays primary.

It also runs two ablations:

  hour-only   occupancy from `hour` alone vs all four sensors, to measure
              rather than assume how much the extra sensors contribute.
  twin-excl   primary test set with the 2.6% of rows that have an exact
              feature twin in training removed, as a sensitivity check.

Anomaly detection uses a ROLLING z-score, not a global one. Phase 2C
established that global z-scores cannot see the stuck-sensor rows (global
z ~ 0.7 and 1.2), so a global detector is not fit for purpose.

Detector parameters are chosen by grid search ON THE TRAINING SPLIT ONLY.
Tuning on the test set would be a quiet form of leakage.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg  # noqa: E402
import preprocessing as pp  # noqa: E402

MODELS_DIR = cfg.PROJECT_ROOT / "models"
OCCUPANCY_MODEL = MODELS_DIR / "occupancy_model.pkl"
TEMPERATURE_MODEL = MODELS_DIR / "temperature_model.pkl"

RULE = "=" * 74
THIN = "-" * 74

# Detector grid. Scored on training data only; the winner is then applied
# unchanged to both test splits.
ZSCORE_GRID = [(w, k) for w in (12, 288) for k in (2.0, 2.5, 3.0)]


# --------------------------------------------------------------------------
# Rolling z-score detector
# --------------------------------------------------------------------------
class RollingZScore:
    """Flag samples that deviate from their own recent local history.

    The baseline for a sample is the mean and standard deviation of the
    PREVIOUS `window` samples only (`shift(1)`). Two consequences:

      * the sample being tested never contributes to its own baseline, so a
        spike cannot inflate the reference it is measured against;
      * no future value is used, so applying the detector to a test period is
        legitimate.

    A FULL window is required before the detector will judge a sample
    (`min_periods == window`). Allowing a short partial window looks harmless
    but is not: with window=288 and min_periods=6 the first day of the series
    was scored against a mean and standard deviation estimated from a handful
    of samples, which produced 43 false positives on day one alone. The
    `window - 1` rows at the start of the series are therefore left unflagged,
    and the count is reported rather than hidden.
    """

    def __init__(self, window: int = 12, threshold: float = 3.0) -> None:
        self.window = window
        self.threshold = threshold
        self.n_warmup = 0
        self.n_zero_variance = 0

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Per-column |z| for each row. Higher means more anomalous."""
        past = frame.shift(1)
        mean = past.rolling(self.window, min_periods=self.window).mean()
        std = past.rolling(self.window, min_periods=self.window).std()
        # A frozen baseline with an exactly repeating value is not an anomaly;
        # a value that departs from a frozen baseline is an infinite deviation.
        zero = std == 0
        self.n_zero_variance = int(zero.to_numpy().sum())
        deviation = (frame - mean).abs()
        z = deviation / std.replace(0, np.nan)
        z = z.where(~zero, np.where(deviation > 0, np.inf, 0.0))
        self.n_warmup = int(z.isna().all(axis=1).sum())
        return z

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """True where any sensor deviates beyond the threshold."""
        z = self.score(frame)
        return (z > self.threshold).any(axis=1).to_numpy()


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2_score(y_true, y_pred),
    }


def _table(title: str, columns: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(str(c)), *(len(str(r[i])) for r in rows)) if rows else len(str(c))
              for i, c in enumerate(columns)]
    print(f"  {'  '.join(str(c).ljust(w) for c, w in zip(columns, widths))}")
    print("  " + "  ".join("-" * w for w in widths))
    for row in rows:
        print("  " + "  ".join(str(v).ljust(w) for v, w in zip(row, widths)))


def _cm_block(metrics: dict[str, float]) -> str:
    return (f"tn={int(metrics['tn'])} fp={int(metrics['fp'])} "
            f"fn={int(metrics['fn'])} tp={int(metrics['tp'])}")


# --------------------------------------------------------------------------
# Occupancy
# --------------------------------------------------------------------------
# Depth grid for the decision tree. The unbounded default reaches ~99.9%
# training accuracy on 1440 rows while generalising worse than a shallow tree,
# so depth and leaf size are selected by time-series cross-validation ON THE
# TRAINING SPLIT ONLY. The test set is never consulted for this choice.
DT_MAX_DEPTHS = (2, 3, 4, 5, 6, 8, 12, None)
DT_MIN_LEAF = (1, 5, 10, 20)
CV_SPLITS = 5


def tune_decision_tree(
    train: pd.DataFrame, features: list[str]
) -> tuple[DecisionTreeClassifier, list[list[str]], dict[str, object]]:
    """Select tree size with TimeSeriesSplit on the training split.

    TimeSeriesSplit rather than KFold, to stay consistent with the rule that
    no fold is scored using a model that saw later data. Returns the fitted
    model, the full search table and the winning parameters.
    """
    X, y, _, _ = pp.build_feature_matrix(train, "occupancy")
    X = X[features]
    splitter = TimeSeriesSplit(n_splits=CV_SPLITS)

    rows: list[list[str]] = []
    best: tuple[float, int | None, int] | None = None
    for max_depth in DT_MAX_DEPTHS:
        for min_leaf in DT_MIN_LEAF:
            scores = []
            for tr_idx, va_idx in splitter.split(X):
                model = DecisionTreeClassifier(
                    random_state=cfg.SEED,
                    max_depth=max_depth,
                    min_samples_leaf=min_leaf,
                )
                model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
                scores.append(
                    f1_score(y.iloc[va_idx], model.predict(X.iloc[va_idx]),
                             zero_division=0)
                )
            mean_f1 = float(np.mean(scores))
            depth_label = "none" if max_depth is None else str(max_depth)
            rows.append([depth_label, str(min_leaf), f"{mean_f1:.4f}"])
            if best is None or mean_f1 > best[0]:
                best = (mean_f1, max_depth, min_leaf)

    assert best is not None
    _, max_depth, min_leaf = best
    model = DecisionTreeClassifier(
        random_state=cfg.SEED, max_depth=max_depth, min_samples_leaf=min_leaf
    )
    model.fit(X, y)
    return model, rows, {"max_depth": max_depth, "min_samples_leaf": min_leaf,
                         "cv_f1": best[0]}


def fit_occupancy(train: pd.DataFrame, features: list[str]) -> DecisionTreeClassifier:
    X, y, _, _ = pp.build_feature_matrix(train, "occupancy")
    X = X[features]
    model = DecisionTreeClassifier(random_state=cfg.SEED)
    model.fit(X, y)
    return model


def occupancy_scores(model: DecisionTreeClassifier, frame: pd.DataFrame,
                     features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X, y, _, _ = pp.build_feature_matrix(frame, "occupancy")
    return y.to_numpy(), model.predict(X[features])


# --------------------------------------------------------------------------
# Temperature
# --------------------------------------------------------------------------
def fit_temperature(train: pd.DataFrame) -> LinearRegression:
    X, y, _, _ = pp.build_feature_matrix(train, "temperature")
    model = LinearRegression()
    model.fit(X, y)
    return model


def temperature_scores(model: LinearRegression, frame: pd.DataFrame
                       ) -> tuple[np.ndarray, np.ndarray]:
    X, y, _, _ = pp.build_feature_matrix(frame, "temperature")
    return y.to_numpy(), model.predict(X)


# --------------------------------------------------------------------------
# Anomaly ground-truth typing (post-hoc reporting only)
# --------------------------------------------------------------------------
def anomaly_type_labels(n_rows: int) -> pd.Series:
    """Recover which anomaly type each row was injected as.

    This reconstructs the Phase 2C injection plan purely so the report can
    state WHICH anomaly types the detector finds and which it misses. It is
    never used as a model input; the detector sees sensor values only, and
    `is_anomaly` is used solely as the evaluation label.
    """
    import data_generator as dg

    _, plan = dg.generate_dataset(return_plan=True)
    labels = pd.Series(["normal"] * n_rows, dtype=object)
    for kind, _column, start, length in plan:
        for i in range(start, min(start + length, n_rows)):
            labels.iloc[i] = kind
    return labels


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> int:
    df = pp.load_dataset()

    primary_train, primary_test = pp.chronological_split(df)
    holdout_train, holdout_test = pp.weekday_holdout_split(df)

    results: dict[str, dict] = {}

    # ---------------------------------------------------------------- split
    print(RULE)
    print("SPLITS")
    print(RULE)
    for name, tr, te in [("primary (chronological)", primary_train, primary_test),
                         ("secondary (weekday_holdout)", holdout_train, holdout_test)]:
        days_tr = sorted({d.day_name() for d in tr["timestamp"]})
        days_te = sorted({d.day_name() for d in te["timestamp"]})
        print(f"  {name}")
        print(f"    train {len(tr):>5} rows  {tr['timestamp'].min()} -> {tr['timestamp'].max()}")
        print(f"      days: {days_tr}")
        print(f"      occupancy {tr['occupancy'].mean() * 100:.2f}%")
        print(f"    test  {len(te):>5} rows  {te['timestamp'].min()} -> {te['timestamp'].max()}")
        print(f"      days: {days_te}")
        print(f"      occupancy {te['occupancy'].mean() * 100:.2f}%")
    print()
    print("  The two splits disagree on difficulty by construction. The primary")
    print("  test period is entirely weekend, which training never sees; the")
    print("  weekday holdout keeps weekends in training. Compare the occupancy")
    print("  scores to separate distribution shift from model weakness.")
    print()

    # ------------------------------------------------------------ occupancy
    full_features = pp.OCCUPANCY_FEATURES
    hour_features = ["hour"]

    dt_primary = fit_occupancy(primary_train, full_features)
    dt_tuned_primary, grid_rows, dt_params = tune_decision_tree(
        primary_train, full_features
    )
    dt_tuned_holdout, _, _ = tune_decision_tree(holdout_train, full_features)
    dt_hour_primary, _, hour_params = tune_decision_tree(
        primary_train, hour_features
    )
    dt_hour_holdout, _, _ = tune_decision_tree(holdout_train, hour_features)

    print(RULE)
    print("OCCUPANCY - Decision Tree Classifier")
    print(RULE)
    print(f"  features        : {full_features}")
    print(f"  target          : occupancy")
    print()
    print("  The unbounded default tree is overfitting:")
    for label, model in [("default (unbounded)", dt_primary), ("tuned", dt_tuned_primary)]:
        a_tr = classification_metrics(
            primary_train["occupancy"].to_numpy(),
            model.predict(primary_train[full_features]),
        )["accuracy"]
        a_te = classification_metrics(
            primary_test["occupancy"].to_numpy(),
            model.predict(primary_test[full_features]),
        )["accuracy"]
        print(f"    {label:<22} depth {model.get_depth():>3}  leaves "
              f"{model.get_n_leaves():>4}  train acc {a_tr:.4f}  test acc {a_te:.4f}")
    print()
    print(f"  Tree size chosen by TimeSeriesSplit({CV_SPLITS}) on the PRIMARY")
    print("  TRAINING SPLIT ONLY - the test set is not consulted:")
    print(f"    max_depth={dt_params['max_depth']}  "
          f"min_samples_leaf={dt_params['min_samples_leaf']}  "
          f"(CV F1 {dt_params['cv_f1']:.4f})")
    print()
    imp = pd.Series(dt_tuned_primary.feature_importances_, index=full_features)
    imp = imp.sort_values(ascending=False)
    print("  feature importance (tuned tree, primary split):")
    for name, value in imp.items():
        bar = "#" * int(round(value * 40))
        print(f"    {name:<14} {value:.4f}  {bar}")
    print()

    occ_rows = []
    for label, model, te, feats in [
        ("tuned all sensors / primary", dt_tuned_primary, primary_test, full_features),
        ("tuned all sensors / weekday_holdout", dt_tuned_holdout, holdout_test, full_features),
        ("default all sensors / primary", dt_primary, primary_test, full_features),
        ("hour only        / primary", dt_hour_primary, primary_test, hour_features),
        ("hour only        / weekday_holdout", dt_hour_holdout, holdout_test, hour_features),
    ]:
        y_true, y_pred = occupancy_scores(model, te, feats)
        m = classification_metrics(y_true, y_pred)
        results[label] = m
        occ_rows.append([
            label,
            f"{m['accuracy']:.4f}", f"{m['precision']:.4f}",
            f"{m['recall']:.4f}", f"{m['f1']:.4f}", _cm_block(m),
        ])
    _table("model / split", ["", "accuracy", "precision", "recall", "F1", "confusion"], occ_rows)
    print()

    # Trivial baselines, because a bare accuracy number is hard to judge.
    base_rows = []
    for label, te in [("primary", primary_test), ("weekday_holdout", holdout_test)]:
        y_occ = te["occupancy"].to_numpy()
        for name, pred in [
            ("always unoccupied", np.zeros_like(y_occ)),
            ("always occupied", np.ones_like(y_occ)),
        ]:
            m = classification_metrics(y_occ, pred)
            base_rows.append([f"{label}: {name}", f"{m['accuracy']:.4f}",
                              f"{m['precision']:.4f}", f"{m['recall']:.4f}",
                              f"{m['f1']:.4f}", _cm_block(m)])
    print("  Trivial baselines for context:")
    _table("split / baseline", ["", "accuracy", "precision", "recall", "F1", "confusion"],
           base_rows)
    print()
    p_base = classification_metrics(
        primary_test["occupancy"].to_numpy(),
        np.zeros(len(primary_test), dtype=int),
    )["accuracy"]
    p_model = results["tuned all sensors / primary"]["accuracy"]
    print(f"    On the primary split the tuned tree scores {p_model:.4f} against a")
    print(f"    'always unoccupied' baseline of {p_base:.4f} - a margin of only")
    print(f"    {p_model - p_base:+.4f}. On weekday_holdout the same model is far")
    print("    clear of baseline, because that split keeps the occupancy rate")
    print("    near 50/50. Quote both, never the primary number alone.")
    print()
    print(f"  hour-only tree selected max_depth={hour_params['max_depth']}, "
          f"min_samples_leaf={hour_params['min_samples_leaf']}")
    print("  via the identical protocol, so the comparison below varies the")
    print("  feature set and not the regularisation.")
    print()

    # --------------------------------------------------------------- ablation
    full_v = results["tuned all sensors / primary"]
    hour_v = results["hour only        / primary"]
    full_h = results["tuned all sensors / weekday_holdout"]
    hour_h = results["hour only        / weekday_holdout"]
    print("  ABLATION - hour only vs all sensors (F1)")
    print(THIN)
    print(f"    primary          : hour only {hour_v['f1']:.4f}  vs  "
          f"all sensors {full_v['f1']:.4f}   "
          f"delta {full_v['f1'] - hour_v['f1']:+.4f}")
    print(f"    weekday_holdout  : hour only {hour_h['f1']:.4f}  vs  "
          f"all sensors {full_h['f1']:.4f}   "
          f"delta {full_h['f1'] - hour_h['f1']:+.4f}")
    print()
    print("    Read the confusion columns above, not just accuracy. `hour only`")
    print("    buys recall by predicting 'occupied' far more often, which is")
    print("    cheap on a 52%-occupied training set and shows up as low")
    print("    precision. F1 is the fairer single number here.")
    print()

    # --------------------------------------------------------- twin ablation
    is_twin, cross_groups, conflicting = pp.cross_boundary_twin_mask(
        primary_train, primary_test
    )
    y_true, y_pred = occupancy_scores(dt_tuned_primary, primary_test, full_features)
    keep = ~is_twin
    m_full = classification_metrics(y_true, y_pred)
    m_kept = classification_metrics(y_true[keep], y_pred[keep])
    print("  ABLATION - twin rows excluded from the primary test set")
    print(THIN)
    print(f"    cross-boundary twin groups : {cross_groups} "
          f"({int(is_twin.sum())} of {len(primary_test)} test rows, "
          f"{is_twin.sum() / len(primary_test) * 100:.1f}%)")
    print(f"    groups with conflicting labels : {conflicting}")
    _table(
        "primary test subset", ["n", "accuracy", "precision", "recall", "F1"],
        [
            ["full test set", len(y_true), f"{m_full['accuracy']:.4f}",
             f"{m_full['precision']:.4f}", f"{m_full['recall']:.4f}", f"{m_full['f1']:.4f}"],
            ["twins excluded", int(keep.sum()), f"{m_kept['accuracy']:.4f}",
             f"{m_kept['precision']:.4f}", f"{m_kept['recall']:.4f}", f"{m_kept['f1']:.4f}"],
            ["delta", "", f"{m_kept['accuracy'] - m_full['accuracy']:+.4f}",
             f"{m_kept['precision'] - m_full['precision']:+.4f}",
             f"{m_kept['recall'] - m_full['recall']:+.4f}",
             f"{m_kept['f1'] - m_full['f1']:+.4f}"],
        ],
    )
    print()
    print("    The dataset is NOT modified by this ablation; rows are only")
    print("    excluded from this evaluation.")
    print()

    # ----------------------------------------------------------- temperature
    lr_primary = fit_temperature(primary_train)
    lr_holdout = fit_temperature(holdout_train)

    print(RULE)
    print("TEMPERATURE - Linear Regression")
    print(RULE)
    print(f"  features        : {pp.TEMPERATURE_FEATURES}")
    print(f"  target          : temperature")
    print("  no temperature_lag1 feature: lag-1 autocorrelation is 0.9106, but")
    print("  adding it would reduce the task to interpolation and inflate R2.")
    print()
    for name, coef in zip(pp.TEMPERATURE_FEATURES, lr_primary.coef_):
        print(f"    coef {name:<14} {coef:+.6f}")
    print(f"    intercept        {lr_primary.intercept_:+.6f}")
    print()
    temp_rows = []
    for label, model, te in [
        ("primary", lr_primary, primary_test),
        ("weekday_holdout", lr_holdout, holdout_test),
    ]:
        y_true, y_pred = temperature_scores(model, te)
        m = regression_metrics(y_true, y_pred)
        results[f"temperature / {label}"] = m
        temp_rows.append([label, len(y_true), f"{m['mae']:.4f}", f"{m['rmse']:.4f}",
                          f"{m['r2']:.4f}",
                          f"mean|err| on mean = {m['mae'] / y_true.mean() * 100:.1f}%"])
    _table("split", ["", "n", "MAE", "RMSE", "R2", "note"], temp_rows)
    print()

    # -------------------------------------------------------------- anomaly
    print(RULE)
    print("ANOMALY - rolling Z-score on sensor columns")
    print(RULE)
    print(f"  inputs          : {pp.ANOMALY_FEATURES}  (is_anomaly is NEVER an input)")
    print(f"  ground truth    : is_anomaly")
    print()

    sensor_cols = pp.ANOMALY_FEATURES
    print("  Grid search on the PRIMARY TRAINING SPLIT ONLY (test untouched):")
    grid_rows = []
    best = None
    for window, threshold in ZSCORE_GRID:
        det = RollingZScore(window, threshold)
        train_flags = det.predict(primary_train[sensor_cols])
        y_tr = primary_train["is_anomaly"].to_numpy()
        m = classification_metrics(y_tr, train_flags)
        grid_rows.append([f"window={window}", f"k={threshold}",
                          f"{m['precision']:.4f}", f"{m['recall']:.4f}",
                          f"{m['f1']:.4f}"])
        if best is None or m["f1"] > best[0]:
            best = (m["f1"], window, threshold)
    _table("config", ["", "", "precision", "recall", "F1"], grid_rows)
    print()
    print(f"  selected on training F1: window={best[1]}, k={best[2]} "
          f"(train F1 {best[0]:.4f})")
    print()

    # Rolling statistics are computed over the full chronological series so the
    # test period inherits real history instead of a cold start, mirroring a
    # detector running continuously on a device. The window looks only
    # backwards, so no future information reaches the test rows.
    detector = RollingZScore(best[1], best[2])
    full_flags = detector.predict(df[sensor_cols])
    test_mask = df["timestamp"] > primary_train["timestamp"].max()
    anomaly_truth = df["is_anomaly"].to_numpy()

    print("  Detector detail:")
    print(f"    window                     : {detector.window} samples "
          f"({detector.window * cfg.SAMPLING_INTERVAL_MINUTES} min)")
    print(f"    threshold                  : {detector.threshold} sigma")
    print(f"    rows without a full window : {detector.n_warmup} (left unflagged)")
    print(f"    zero-variance window cells : {detector.n_zero_variance}")
    print()

    anom_rows = []
    subsets = [
        ("primary", df[test_mask]),
        ("weekday_holdout", holdout_test),
    ]
    for label, te in subsets:
        idx = te.index.to_numpy()
        m = classification_metrics(df["is_anomaly"].to_numpy()[idx], full_flags[idx])
        results[f"anomaly / {label}"] = m
        anom_rows.append([label, len(idx), f"{m['precision']:.4f}", f"{m['recall']:.4f}",
                          f"{m['f1']:.4f}", _cm_block(m)])
    _table("split", ["", "n", "precision", "recall", "F1", "confusion"], anom_rows)
    print()

    # Per-type recovery, for the report only.
    types = anomaly_type_labels(len(df))
    primary_types = types[test_mask.to_numpy()].reset_index(drop=True)
    primary_flags = pd.Series(full_flags[test_mask.to_numpy()].astype(int),
                              index=primary_types.index)
    detected = (
        pd.DataFrame({"type": primary_types, "flag": primary_flags.to_numpy()})
        .groupby("type")
        .agg(rows=("flag", "size"), detected=("flag", "sum"))
    )
    detected["recall"] = detected["detected"] / detected["rows"]
    print("  Per-anomaly-type recovery on the primary test set:")
    _table("type", ["rows", "detected", "recall"],
           [[i, int(r["rows"]), int(r["detected"]), f"{r['recall']:.2f}"]
            for i, r in detected.iterrows()])
    print()
    anomaly_types = detected.drop(index="normal", errors="ignore")
    total_anom = int(anomaly_types["rows"].sum())
    total_found = int(anomaly_types["detected"].sum())
    print(f"  Overall on the primary test set: {total_found}/{total_anom} "
          f"injected anomalies detected ({total_found / total_anom * 100:.1f}%)")
    print()

    # The two evaluation splits together cover only 16 of the 40 injected
    # anomalies, so a dataset-wide figure is the honest summary of what this
    # detector can actually do.
    scorable = df.index[df.index >= detector.window].to_numpy()
    wide_truth = anomaly_truth[scorable]
    wide_found = int((full_flags[scorable] & (wide_truth == 1)).sum())
    wide_total = int((wide_truth == 1).sum())
    wide_fp = int((full_flags[scorable] & (wide_truth == 0)).sum())
    print(f"  Dataset-wide, excluding the {detector.n_warmup} warmup rows:")
    print(f"    detected {wide_found}/{wide_total} injected anomalies "
          f"({wide_found / wide_total * 100:.1f}%), "
          f"{wide_fp} false positives in {len(scorable)} rows")
    print()
    missed = [i for i, r in anomaly_types.iterrows() if r["recall"] < 0.5]
    if missed:
        print(f"  Types the rolling z-score largely misses: {missed}")
        print("  A stuck sensor holds a constant value, so once the rolling")
        print("  window fills with repeats the local standard deviation is")
        print("  zero and the deviation from the baseline is also zero. A")
        print("  rolling z-score cannot detect this class by construction;")
        print("  a flatline/variance-collapse rule is required (Phase 5).")
    print()
    print("  CAUTION on the headline F1 above: the primary test period happens")
    print("  to contain only spike/drop anomalies, which any z-score catches,")
    print("  and just 7 of them. A perfect score on n=7 is a small-sample")
    print("  result, not evidence of a strong detector. The weekday_holdout")
    print("  column, whose anomalies include stuck runs, is the fairer read.")
    print()


    # --------------------------------------------------------------- persist
    print(RULE)
    print("MODEL PERSISTENCE")
    print(RULE)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(dt_tuned_primary, OCCUPANCY_MODEL)
    joblib.dump(lr_primary, TEMPERATURE_MODEL)
    for path in (OCCUPANCY_MODEL, TEMPERATURE_MODEL):
        size = path.stat().st_size
        print(f"  wrote models/{path.name}  ({size} bytes)")
    print()
    print(f"  occupancy_model.pkl  : tuned tree, max_depth="
          f"{dt_params['max_depth']}, min_samples_leaf="
          f"{dt_params['min_samples_leaf']}, features {full_features}")
    print("  temperature_model.pkl: LinearRegression, "
          f"features {pp.TEMPERATURE_FEATURES}")
    print("  Both were fitted on the primary training split only")
    print(f"  ({len(primary_train)} rows, {primary_train['timestamp'].min()} ->")
    print(f"  {primary_train['timestamp'].max()}). No model was refitted on the")
    print("  full dataset, and the anomaly detector has no fitted state - it is")
    print(f"  fully described by window={detector.window}, k={detector.threshold}.")
    print()

    # ------------------------------------------------------------- integrity
    print(RULE)
    print("SOURCE DATA INTEGRITY")
    print(RULE)
    print(f"  rows read        : {len(df)}")
    print(f"  is_anomaly sum   : {int(df['is_anomaly'].sum())} "
          f"(ground truth, used only as a label)")
    print(f"  occupancy sum    : {int(df['occupancy'].sum())}")
    print("  data/sensor_data.csv was opened read-only and not rewritten.")
    print(RULE)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
