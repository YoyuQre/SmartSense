"""
Phase 3 - exploratory data analysis and the human-readable report.

Run:  python src/eda.py

Produces the eight EDA figures under visualizations/ and prints the
data-quality / distribution / temporal / split report required by Phase 3.

Reads data/sensor_data.csv through src/preprocessing.py and never writes to
the dataset. No model is trained here; the only outputs are tables, figures
and the split definition that Phase 4 will consume.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display on the build machine

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg  # noqa: E402
import preprocessing as pp  # noqa: E402

FIG_DIR = cfg.VISUALIZATIONS_DIR
RULE = "=" * 74
THIN = "-" * 74


def _save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def plot_sensor_time_series(df: pd.DataFrame) -> None:
    """All three sensors over the full week.

    Stacked subplots sharing the x axis rather than three overlaid lines:
    temperature (~20-32), light (0-1000) and motion (0/1) span three orders of
    magnitude, and a shared axis would flatten the motion trace into the axis.
    """
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)

    axes[0].plot(df["timestamp"], df["temperature"], lw=0.8, color="tab:red")
    axes[0].axhline(cfg.HVAC_COOLING_THRESHOLD, ls="--", lw=1, color="grey",
                    label=f"HVAC threshold {cfg.HVAC_COOLING_THRESHOLD} C")
    axes[0].set_ylabel("Temperature (C)")
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].plot(df["timestamp"], df["light_level"], lw=0.8, color="tab:orange")
    axes[1].axhline(cfg.LIGHT_THRESHOLD, ls="--", lw=1, color="grey",
                    label=f"Light threshold {cfg.LIGHT_THRESHOLD}")
    axes[1].set_ylabel("Light level (0-1000)")
    axes[1].legend(loc="upper right", fontsize=8)

    axes[2].step(df["timestamp"], df["motion"], where="post", lw=0.9,
                 color="tab:blue", label="motion")
    axes[2].step(df["timestamp"], df["occupancy"], where="post", lw=0.9,
                 color="tab:green", alpha=0.6, label="occupancy")
    axes[2].set_ylabel("Binary")
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].legend(loc="upper right", fontsize=8, ncol=2)
    axes[2].set_xlabel("Timestamp")

    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle("Sensor time series - 7 days at 5-minute intervals")
    _save(fig, "sensor_time_series.png")


def plot_occupancy_distribution(df: pd.DataFrame) -> None:
    occupied = int(df["occupancy"].sum())
    unoccupied = int((df["occupancy"] == 0).sum())
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(["Unoccupied", "Occupied"], [unoccupied, occupied],
                  color=["tab:gray", "tab:green"])
    for bar, value in zip(bars, [unoccupied, occupied]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 12,
                f"{value}\n({value / len(df) * 100:.1f}%)", ha="center")
    ax.set_ylabel("Samples")
    ax.set_title("Occupancy distribution")
    ax.set_ylim(0, max(unoccupied, occupied) * 1.22)
    ax.grid(alpha=0.25, axis="y")
    _save(fig, "occupancy_distribution.png")


def plot_motion_vs_occupancy(df: pd.DataFrame) -> None:
    """Contingency of motion against occupancy.

    The point is that they are related but not identical: a model that read
    `motion` as a copy of `occupancy` would be learning nothing.
    """
    ct = pd.crosstab(df["motion"], df["occupancy"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    im = axes[0].imshow(ct.values, cmap="Blues", vmin=0, vmax=ct.values.max())
    axes[0].set_xticks([0, 1], ["occupancy=0", "occupancy=1"])
    axes[0].set_yticks([0, 1], ["motion=0", "motion=1"])
    for i in range(2):
        for j in range(2):
            value = int(ct.values[i, j])
            axes[0].text(j, i, f"{value}\n{value / len(df) * 100:.1f}%",
                         ha="center", va="center")
    axes[0].set_title("Contingency (rows = motion, cols = occupancy)")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    agreement = float((df["motion"] == df["occupancy"]).mean())
    labels = ["no motion\n+ empty", "no motion\n+ occupied", "motion\n+ empty",
              "motion\n+ occupied"]
    values = [int(ct.values[0, 0]), int(ct.values[0, 1]),
              int(ct.values[1, 0]), int(ct.values[1, 1])]
    colors = ["lightgrey", "tab:green", "tab:orange", "tab:blue"]
    bars = axes[1].bar(labels, values, color=colors)
    for bar, value in zip(bars, values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                     str(value), ha="center", fontsize=9)
    axes[1].set_title(f"motion vs occupancy - agree on {agreement * 100:.1f}% of rows")
    axes[1].set_ylabel("Samples")
    axes[1].tick_params(axis="x", labelsize=8)
    axes[1].grid(alpha=0.25, axis="y")

    fig.tight_layout()
    _save(fig, "motion_vs_occupancy.png")


def plot_light_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(df["light_level"], bins=50, color="tab:orange", edgecolor="white")
    ax.axvline(cfg.LIGHT_THRESHOLD, ls="--", color="k", lw=1.2,
               label=f"light threshold {cfg.LIGHT_THRESHOLD}")
    ax.axvline(df["light_level"].mean(), ls=":", color="tab:blue", lw=1.5,
               label=f"mean {df['light_level'].mean():.1f}")
    ax.set_xlabel("Light level (normalised 0-1000, NOT calibrated lux)")
    ax.set_ylabel("Samples")
    ax.set_title("Light level distribution")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    _save(fig, "light_distribution.png")


def plot_temperature_time_series(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 4.5))
    normal = df[df["is_anomaly"] == 0]
    anomalous = df[df["is_anomaly"] == 1]
    ax.plot(normal["timestamp"], normal["temperature"], lw=0.8, color="tab:red",
            alpha=0.75, label="normal")
    ax.scatter(anomalous["timestamp"], anomalous["temperature"], s=18,
               color="black", marker="v", zorder=5,
               label=f"anomalous (n={len(anomalous)})")
    ax.axhline(cfg.HVAC_COOLING_THRESHOLD, ls="--", color="grey", lw=1,
               label=f"HVAC threshold {cfg.HVAC_COOLING_THRESHOLD} C")
    ax.set_ylabel("Temperature (C)")
    ax.set_xlabel("Timestamp")
    ax.set_title("Temperature over 7 days")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    _save(fig, "temperature_time_series.png")


def plot_temperature_by_hour(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    data = [df.loc[df["hour"] == h, "temperature"].to_numpy() for h in range(24)]
    bp = ax.boxplot(data, positions=range(24), widths=0.6, showfliers=False)
    means = [d.mean() for d in data]
    ax.plot(range(24), means, "o-", color="tab:red", lw=1.2, ms=4,
            label="hourly mean")
    ax.axhline(cfg.HVAC_COOLING_THRESHOLD, ls="--", color="grey", lw=1,
               label=f"HVAC threshold {cfg.HVAC_COOLING_THRESHOLD} C")
    ax.set_xticks(range(24))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Temperature (C)")
    ax.set_title("Temperature distribution by hour (box = IQR, line = mean)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    _save(fig, "temperature_by_hour.png")


def plot_occupancy_by_hour(df: pd.DataFrame, profile: pd.DataFrame) -> None:
    """Occupancy rate per hour, with the two single-class hours called out.

    Hours 2 and 4 are unoccupied in all 84 samples, so they contribute no
    occupancy signal. They are highlighted rather than dropped, because
    hiding them would misrepresent how much `hour` alone can explain.
    """
    fig, ax = plt.subplots(figsize=(11, 4.5))
    rates = profile["occupancy_rate"].to_numpy()
    colors = ["tab:red" if not mixed else "tab:green"
              for mixed in profile["is_mixed"].to_numpy()]
    ax.bar(range(24), rates, color=colors)

    for hour, row in profile.iterrows():
        if not row["is_mixed"]:
            ax.annotate(
                f"single-class\n{int(row['occupied'])}/{int(row['samples'])}",
                xy=(hour, 0), xytext=(hour, 0.22), ha="center", fontsize=7,
                arrowprops=dict(arrowstyle="->", lw=0.8, color="tab:red"),
                color="tab:red",
            )

    ax.axhline(df["occupancy"].mean(), ls="--", color="k", lw=1,
               label=f"overall rate {df['occupancy'].mean() * 100:.1f}%")
    ax.set_xticks(range(24))
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Occupancy rate")
    ax.set_title(
        f"Occupancy by hour - {int(profile['is_mixed'].sum())}/24 hours mixed "
        "(red = single-class, no signal)"
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25, axis="y")
    _save(fig, "occupancy_by_hour.png")


def plot_anomaly_overview(df: pd.DataFrame) -> None:
    anomalous = df[df["is_anomaly"] == 1]
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    axes[0].plot(df["timestamp"], df["temperature"], lw=0.7, color="lightgrey")
    axes[0].scatter(anomalous["timestamp"], anomalous["temperature"], s=26,
                    color="tab:red", zorder=5, label="is_anomaly = 1")
    axes[0].set_ylabel("Temperature (C)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].plot(df["timestamp"], df["light_level"], lw=0.7, color="lightgrey")
    axes[1].scatter(anomalous["timestamp"], anomalous["light_level"], s=26,
                    color="tab:red", zorder=5, label="is_anomaly = 1")
    axes[1].set_ylabel("Light level")
    axes[1].set_xlabel("Timestamp")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

    fig.suptitle(
        f"Anomalous samples against normal - {len(anomalous)} of {len(df)} rows "
        f"({len(anomalous) / len(df) * 100:.2f}%)"
    )
    _save(fig, "anomaly_overview.png")


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def print_data_quality(df: pd.DataFrame) -> None:
    print(RULE)
    print("DATASET")
    print(RULE)
    print(f"  rows            : {len(df)}")
    print(f"  columns         : {len(df.columns)}  ({', '.join(df.columns)})")
    print(f"  date range      : {df['timestamp'].min()} -> {df['timestamp'].max()}")
    steps = sorted({int(d.total_seconds()) for d in df['timestamp'].diff().dropna()})
    print(f"  interval        : {steps} seconds "
          f"({'uniform' if len(steps) == 1 else 'IRREGULAR'})")
    print()

    print(RULE)
    print("MISSING VALUES")
    print(RULE)
    print(f"  {'column':<14}{'missing':>10}{'pct':>10}")
    for col in df.columns:
        miss = int(df[col].isna().sum())
        print(f"  {col:<14}{miss:>10}{miss / len(df) * 100:>9.2f}%")
    print()

    print(RULE)
    print("NUMERIC STATISTICS")
    print(RULE)
    print(f"  {'column':<14}{'min':>10}{'max':>10}{'mean':>10}"
          f"{'median':>10}{'std':>10}")
    for col in ["motion", "light_level", "temperature", "occupancy", "is_anomaly"]:
        s = df[col]
        print(f"  {col:<14}{s.min():>10.2f}{s.max():>10.2f}{s.mean():>10.2f}"
              f"{s.median():>10.2f}{s.std():>10.3f}")
    print()

    print(RULE)
    print("BINARY DISTRIBUTIONS")
    print(RULE)
    for col in ["occupancy", "motion", "is_anomaly"]:
        ones = int(df[col].sum())
        zeros = int(len(df) - ones)
        print(f"  {col:<12} 0 = {zeros:>5} ({zeros / len(df) * 100:>5.2f}%)   "
              f"1 = {ones:>5} ({ones / len(df) * 100:>5.2f}%)")
    print()


def print_temporal(df: pd.DataFrame, profile: pd.DataFrame) -> None:
    print(RULE)
    print("TEMPORAL STRUCTURE")
    print(RULE)
    lags = pp.autocorrelation_report(df, "temperature", (1, 12, 288))
    meaning = {1: "5 minutes", 12: "1 hour", 288: "1 day"}
    for lag, value in lags.items():
        print(f"  temperature lag-{lag:<3} ({meaning[lag]:<9}) : {value:.6f}")
    print()
    print("  Light level, for contrast (a daylight cycle, not a random walk):")
    light_lags = pp.autocorrelation_report(df, "light_level", (1, 12, 288))
    for lag, value in light_lags.items():
        print(f"  light_level lag-{lag:<3} ({meaning[lag]:<9}) : {value:.6f}")
    print()
    print(THIN)
    print("  CORRECTION to the Phase 2C report: it quoted lag-1 ~= 0.999,")
    print("  which was measured on the smooth series BEFORE anomaly")
    print("  injection. The shipped CSV contains 40 injected anomalies")
    print("  (spikes, drops, frozen runs) which break local smoothness.")
    print("  Measured on the file Phase 4 will actually train on, the real")
    print("  value is the 0.910602 above. Still far too high for a random")
    print("  split, but 0.999 must not be quoted as a property of the data.")
    print()
    print("  Note lag-288 (0.903) exceeds lag-12 (0.869): autocorrelation does")
    print("  not decay monotonically because the series is dominated by a daily")
    print("  cycle. 'Same time yesterday' is more similar than 'one hour ago'.")
    print()
    print(THIN)
    print("  Implication: a random train/test split would put a sample's")
    print("  immediate neighbour in the training set. Phase 3 therefore uses")
    print("  a chronological split only. See SPLIT below.")
    print()

    print(RULE)
    print("OCCUPANCY BY HOUR")
    print(RULE)
    mixed = int(profile["is_mixed"].sum())
    single = [int(h) for h in profile.index[~profile["is_mixed"]]]
    print(f"  mixed hours     : {mixed}/24")
    print(f"  single-class    : {single if single else 'none'}")
    if single:
        for hour in single:
            row = profile.loc[hour]
            print(f"    hour {hour:>2}       : "
                  f"{int(row['occupied'])}/{int(row['samples'])} occupied")
        print()
        print("  These hours carry no occupancy signal. `hour` is kept as a")
        print("  feature anyway - 22 hours remain informative - but `hour`")
        print("  alone must not be expected to separate classes.")
    print()


def print_split_and_leakage(df: pd.DataFrame) -> None:
    train, test = pp.chronological_split(df)
    info = pp.describe_split(train, test)

    print(RULE)
    print("CHRONOLOGICAL SPLIT (no train_test_split)")
    print(RULE)
    print(f"  train rows      : {info['train_rows']}")
    print(f"  test rows       : {info['test_rows']}")
    print(f"  train period    : {info['train_start']} -> {info['train_end']}")
    print(f"  test period     : {info['test_start']} -> {info['test_end']}")
    print(f"  train occ rate  : {info['train_occupancy_rate'] * 100:.2f}%")
    print(f"  test  occ rate  : {info['test_occupancy_rate'] * 100:.2f}%")
    print(f"  train anom rate : {info['train_rate'] * 100:.2f}%")
    print(f"  test  anom rate : {info['test_rate'] * 100:.2f}%")
    print()

    # This is the most important caveat in Phase 3. The dataset deliberately
    # starts on a Monday, so "first 5 days / last 2 days" lines up exactly
    # with weekday / weekend. The model never sees a weekend example during
    # training, so a Phase 4 accuracy drop would reflect that shift rather
    # than a weak model.
    train_days = sorted({d.day_name() for d in train["timestamp"]})
    test_days = sorted({d.day_name() for d in test["timestamp"]})
    weekend_train = [d for d in train_days if d in ("Saturday", "Sunday")]
    print(RULE)
    print("*** DISTRIBUTION SHIFT WARNING ***")
    print(RULE)
    print(f"  train day names : {train_days}")
    print(f"  test  day names : {test_days}")
    print(f"  weekend days in train : {len(weekend_train)}")
    print()
    print("  The dataset starts on a Monday, so a 5/2 chronological split")
    print("  puts all 5 weekdays in training and both weekend days in test.")
    print(f"  Occupancy falls from {info['train_occupancy_rate'] * 100:.1f}% to "
          f"{info['test_occupancy_rate'] * 100:.1f}%,")
    print("  so the model will over-predict occupancy on the test period.")
    print()
    print("  This split was specified for Phase 3 and is implemented as")
    print("  requested, but Phase 4 should ALSO report a second split that")
    print("  holds out one weekday and trains on everything else, to separate")
    print("  'the model is bad' from 'the test period is all weekend'.")
    print()

    print(RULE)
    print("LEAKAGE CHECKS")
    print(RULE)
    checks = pp.leakage_checks(train, test, source=df)
    for check in checks:
        print("  " + check.render())
    print()

    print(RULE)
    print("FEATURE DEFINITIONS (for Phase 4)")
    print(RULE)
    for task, (features, target) in pp.TASKS.items():
        print(f"  {task:<12} X = {features}")
        print(f"  {'':<12} y = {target}")
    print()
    print("  Excluded from every feature set:")
    print("    timestamp  - an index, not a measurement; would let a tree")
    print("                 memorise absolute dates.")
    print("    is_anomaly - ground truth recorded at injection time; using it")
    print("                 as an input is direct answer leakage.")
    print()
    print("  Scaling: deliberately none. A decision tree is invariant to")
    print("  feature scale, and linear regression on hour/light/temperature")
    print("  is well conditioned without it.")
    print()

    failed = [c for c in pp.validate_dataset(df) if not c.passed]
    failed += [c for c in checks if not c.passed]
    print(RULE)
    if failed:
        print(f"RESULT: {len(failed)} CHECK(S) FAILED")
        for c in failed:
            print("  " + c.render())
    else:
        print(f"RESULT: ALL {len(checks) + len(pp.validate_dataset(df))} "
              "DATA QUALITY + LEAKAGE CHECKS PASSED")
    print(RULE)


def main() -> int:
    df = pp.load_dataset()
    profile = pp.hourly_occupancy_profile(df)

    print_data_quality(df)
    print_temporal(df, profile)

    print(RULE)
    print("FIGURES")
    print(RULE)
    plot_sensor_time_series(df)
    plot_occupancy_distribution(df)
    plot_motion_vs_occupancy(df)
    plot_light_distribution(df)
    plot_temperature_time_series(df)
    plot_temperature_by_hour(df)
    plot_occupancy_by_hour(df, profile)
    plot_anomaly_overview(df)
    print()

    print_split_and_leakage(df)

    failed = [c for c in pp.validate_dataset(df) if not c.passed]
    train, test = pp.chronological_split(df)
    failed += [c for c in pp.leakage_checks(train, test, source=df) if not c.passed]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
