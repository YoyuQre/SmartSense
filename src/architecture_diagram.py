"""Phase 7 - render the labeled 5-layer architecture diagram.

Produces docs/architecture.png. Layers follow the required report order:
sensor -> connectivity -> processing -> AI model -> application.

Run:  python src/architecture_diagram.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = ROOT / "docs" / "architecture.png"

# palette: one hue per layer, dark text on light fills
LAYERS = [
    dict(
        name="1  SENSOR LAYER  (on ESP32)",
        edge="#1b5e20",
        fill="#e8f5e9",
        nodes=[
            ("PIR Motion\nGPIO 27", "digital IN"),
            ("LDR Photoresistor\nGPIO 34 (ADC1_CH6)", "analog IN, 0-1000"),
            ("DS18B20\nGPIO 4 (1-Wire)", "temperature"),
            ("Light LED\nGPIO 18", "OUT"),
            ("HVAC LED\nGPIO 19", "OUT"),
        ],
        note="5-minute sampling  |  24 samples/hour  |  light_level is a normalised 0-1000 scale, not lux",
    ),
    dict(
        name="2  CONNECTIVITY LAYER",
        edge="#004d40",
        fill="#e0f2f1",
        nodes=[
            ("UART Serial Link\n115200 baud", "ESP32 -> host"),
            ("CSV Replay Files\nfrozen, SHA-256 pinned", "offline transport"),
        ],
        note="no cloud dependency  |  no network protocol",
    ),
    dict(
        name="3  PROCESSING LAYER  (host)",
        edge="#1a237e",
        fill="#e8eaf6",
        nodes=[
            ("Validation\nschema + ranges", "preprocessing"),
            ("Occupancy\nDecisionTree\ndepth 3", "inference"),
            ("Temperature\nLinearRegression", "inference"),
            ("Anomaly Detection\nrolling Z + flatline", "detection.py"),
            ("Control Logic\nlight + HVAC + fail-safe", "control_logic.py"),
        ],
        note="models loaded from .pkl  |  inference only, never retrained  |  MCU runs thresholds, not models",
    ),
    dict(
        name="4  AI MODEL LAYER  (host)",
        edge="#4a148c",
        fill="#f3e5f5",
        nodes=[
            ("occupancy_model.pkl\nF1 0.694", "primary"),
            ("temperature_model.pkl\nRMSE 1.95 C", "primary"),
            ("Sensor Health\nnormal / suspect", "flatline only"),
        ],
        note="SHA-256 pinned  |  Phase 2 / 4 artifacts unchanged",
    ),
    dict(
        name="5  APPLICATION LAYER  (host)",
        edge="#bf360c",
        fill="#fbe9e7",
        nodes=[
            ("Streamlit Dashboard\nSystem Status", "app.py"),
            ("ML Performance\nPR / confusion", "app.py"),
            ("Sensor & Anomaly\nmonitoring", "app.py"),
            ("Intelligent Control\ndecisions", "app.py"),
            ("Historical Simulation\n(default mode)", "replay"),
        ],
        note="reproducible from frozen files  |  serial mode gated on real hardware",
    ),
]

FIG_W = 17.0
ROW_H = 2.05
BAND_H = 1.34
TOP_PAD = 0.55
BOT_PAD = 0.75
GAP = 0.40


def main() -> int:
    n = len(LAYERS)
    fig_h = TOP_PAD + n * (ROW_H + GAP) + BOT_PAD
    fig, ax = plt.subplots(figsize=(FIG_W, fig_h))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Map data coordinates onto the axes with a top-down origin.
    ax.set_xlim(0, 100)
    ax.set_ylim(fig_h, 0)

    y = TOP_PAD
    bands = []
    for layer in LAYERS:
        bands.append((y, layer))
        y += ROW_H + GAP

    for idx, (by, layer) in enumerate(bands):
        nw = len(layer["nodes"])
        # leave a label column on the left
        label_w = 15.0
        x0 = label_w
        span = 100 - x0 - 1.0
        pad = 0.55
        w = (span - pad * (nw - 1)) / nw

        # band background
        ax.add_patch(
            FancyBboxPatch(
                (0.4, by), 99.2, BAND_H,
                boxstyle="round,pad=0.02,rounding_size=0.5",
                linewidth=0.0, facecolor=layer["fill"], alpha=0.55, zorder=0,
            )
        )
        # band label
        ax.text(
            0.9, by + BAND_H / 2, layer["name"],
            ha="left", va="center", fontsize=11.5, fontweight="bold",
            color=layer["edge"], zorder=3,
        )

        node_y = by + BAND_H + 0.30
        for i, (title, sub) in enumerate(layer["nodes"]):
            nx = x0 + i * (w + pad)
            ax.add_patch(
                FancyBboxPatch(
                    (nx, node_y), w, ROW_H - 0.55,
                    boxstyle="round,pad=0.02,rounding_size=0.45",
                    linewidth=1.5, edgecolor=layer["edge"],
                    facecolor="white", zorder=2,
                )
            )
            ax.text(
                nx + w / 2, node_y + (ROW_H - 0.55) / 2 + 0.16, title,
                ha="center", va="center", fontsize=8.6, fontweight="bold",
                color="#1a1a1a", zorder=3, linespacing=1.45,
            )
            ax.text(
                nx + w / 2, node_y + 0.20, sub,
                ha="center", va="center", fontsize=7.4, style="italic",
                color="#666666", zorder=3,
            )

        ax.text(
            x0, by + BAND_H / 2, layer["note"],
            ha="left", va="center", fontsize=7.6, color=layer["edge"], zorder=3,
        )

        # arrow to next band
        if idx < n - 1:
            cx = 50
            y0 = node_y + (ROW_H - 0.55) + 0.06
            y1 = y0 + GAP - 0.02
            ax.add_patch(
                FancyArrowPatch(
                    (cx, y0), (cx, y1),
                    arrowstyle="-|>", mutation_scale=17,
                    linewidth=1.7, color="#455a64", zorder=4,
                )
            )
            ax.text(
                cx + 1.2, (y0 + y1) / 2,
                "reads" if idx == 0 else ("scores" if idx == 1 else ("informs" if idx == 2 else "drives")),
                ha="left", va="center", fontsize=7.2, color="#455a64", zorder=4,
            )

    # The y-axis is inverted (set_ylim(fig_h, 0)), so small y is the TOP.
    ax.text(
        0.4, 0.28,
        "Occupancy-Based Lighting/HVAC",
        ha="left", va="center", fontsize=12.5, fontweight="bold", color="#0d47a1",
    )
    ax.text(
        99.6, 0.28,
        "Team 8  |  241829, 241833, 241836",
        ha="right", va="center", fontsize=9.0, color="#455a64",
    )

    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT.relative_to(ROOT)}  ({OUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
