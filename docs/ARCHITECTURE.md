# System Architecture

**Project:** Occupancy-Based Lighting/HVAC
Using Machine Learning
**Team:** 8 — 241829 Mohammed Amin Kaifi Patel, 241833 Mohammed Yahya Mohammed
Qayyum Qureshi, 241836 Abuzar Sayyed

![System architecture](architecture.png)

Regenerate with `python src/architecture_diagram.py` → `docs/architecture.png`.

The diagram follows the five layers the report format requires: **sensor →
connectivity → processing → AI model → application**. Each layer is described
below, followed by the data contract that crosses each boundary.

---

## Layer 1 — Sensor layer (ESP32 DevKit C V4)

| Device | Role | Pin | Mode |
| --- | --- | --- | --- |
| PIR motion sensor | motion / presence | GPIO 27 | digital input |
| LDR module | ambient light | GPIO 34 (ADC1_CH6) | analog input |
| DS18B20 | temperature | GPIO 4 | 1-Wire |
| Light LED indicator | lighting state | GPIO 18 | digital output |
| HVAC LED indicator | HVAC state | GPIO 19 | digital output |

Defined in `src/main.cpp` lines 20–25. GPIO 34 is input-only on the ESP32, so
no pull-up is configured for it; the LDR module carries its own divider.
`DS18B20_PIN` drives a `OneWire` bus and the firmware prints the number of
devices found at boot, so a missing probe is visible in the serial log rather
than silently returning 0 °C. An invalid DS18B20 reading is reported as
`-127.00`, logged on its own `ERR` line, and never used to drive HVAC.

**Units.** `light_level` is a **normalised 0–1000 scale derived by `map()` from
the raw ADC count — it is not calibrated lux.** The firmware states this in its
own boot banner (`src/main.cpp:81`). Throughout this project the value is
written as "light_level" or "0–1000 units", never "lux", because a real LDR
transfer function would need calibration against a reference instrument. The
300 lighting threshold is therefore a threshold on that normalised scale.

Sampling interval is **5 minutes**, fixed in `src/config.py`
(`SAMPLING_INTERVAL_MINUTES = 5`). Every downstream threshold in the project is
expressed in samples *and* minutes for this reason, so a threshold cannot be
misread when the interval changes.

**The dataset that feeds the models is not read from these pins at run time.**
`data/sensor_data.csv` is produced by `src/data_generator.py`, a simulation of
this exact sampling contract. That boundary is deliberate and is restated in
`docs/LIMITATIONS.md`.

## Layer 2 — Connectivity layer

Two transports, no cloud dependency and no network protocol:

1. **UART serial, 115200 baud** — live firmware → host. The firmware prints a
   CSV header at boot, then one line every 5 minutes:

   ```
   DATA,<millis>,<motion>,<light_level>,<temperature>,<light_status>,<hvac_status>
   ```

   plus `[INIT]` diagnostics at boot and `ERR,TEMP_INVALID,raw=...` on a bad
   probe read. The protocol is specified in `README.md` §8 and parsed by
   `src/dashboard_data.py::parse_serial_lines`.
2. **Frozen CSV files** — `data/sensor_data.csv` and
   `data/control_output.csv`, both SHA-256 pinned. Every number in the dashboard
   and in the report is reproducible from these two files with no hardware
   attached. This is what makes the results checkable rather than asserted.

### What the firmware does and does not do

Worth stating precisely, because it is the honest boundary of this project:

- The ESP32 **does not run the ML models.** It reads sensors, applies a fixed
  threshold, and drives the indicator LEDs. The models run on the host, in
  Python.
- The firmware's occupancy test is **raw PIR motion**
  (`main.cpp:135`: `motion == 1 && temperatureValid && temperature > 26.0`),
  not the DecisionTree. It is a stand-in for the model, so the firmware alone
  cannot demonstrate the ML result.
- The firmware has **no heating branch** and reports a single boolean `hvacOn`.
  The Python control layer returns a three-valued `OFF / COOLING / HEATING`
  field. Heating therefore exists in the analysis layer and over the replay
  output, but **is not yet expressible on the wire**. See
  `docs/LIMITATIONS.md`.

## Layer 3 — Processing layer

Pure, testable Python. No hidden state, no I/O in the decision functions.

| Module | Responsibility |
| --- | --- |
| `src/preprocessing.py` | schema + range validation, `hour` feature, chronological split |
| `src/modeling.py` | tuning grid, fits, scores, confusion matrices |
| `src/detection.py` | rolling Z-score + range-based flatline rules, sensor health |
| `src/control_logic.py` | lighting and HVAC rules, fail-safe gating |
| `src/phase5_run.py` | historical replay driver → `data/control_output.csv` |
| `src/dashboard_data.py` | inference-only metrics, filters, serial parsing |

**Split policy.** `chronological_split()` cuts the 2016 rows at a single
timestamp: 1440 train rows (2026-01-05 → 01-09), 576 test rows
(2026-01-10 → 01-11). This is deliberately *not* `train_test_split`. Measured
lag-1 autocorrelation of temperature is **0.90**, so a random split puts a
test row's immediate neighbour in the training set and hands the model its own
answer. `max(train) < min(test)` holds by construction.

**Occupancy features:** `hour`, `motion`, `light_level`, `temperature`.
**Temperature features:** `hour`, `motion`, `light_level`, `occupancy`.
In both cases the target is excluded from its own features.

## Layer 4 — AI model layer

| Model | File | Shape | Purpose |
| --- | --- | --- | --- |
| Occupancy | `models/occupancy_model.pkl` | `DecisionTreeClassifier`, depth 3, min-leaf 20, 8 leaves | binary occupancy |
| Temperature | `models/temperature_model.pkl` | `LinearRegression`, 4 features | temperature forecast |

**Inference only.** The dashboard and the replay load these two `.pkl` files and
call `predict`. They never call `fit`, and `src/final_audit.py` fails the build
if a training API appears in any shipped path. Retraining would silently
invalidate every recorded metric, so it is treated as a build error.

**Integrity checking.** The SHA-256 values live in `FROZEN_SHA256` in
`src/dashboard_data.py`, not in `config.py`. `verify_frozen_files()` re-hashes
each artifact and reports `unchanged` / `CHANGED` / `MISSING`, and the result is
rendered on the System Status tab. It **reports rather than raises**, so an
examiner can see the drift in the UI instead of being met with a stack trace.
A changed hash is a visible warning, not a crash.

## Layer 5 — Application layer

`app.py`, a Streamlit dashboard with four tabs:

1. **System Status** — dataset size, model identities, artifact hashes.
2. **ML Performance** — precision/recall/F1 for occupancy, MAE/RMSE/R² for
   temperature, both recomputed by inference at load time.
3. **Sensor & Anomaly Monitoring** — time series with anomaly markers, and
   sensor-health events.
4. **Intelligent Control** — the `light_status` / `hvac_status` decisions with
   the reason that produced each one.

Plus a **mode switch**, and this is the part that matters for honesty:

- **Historical Simulation** — the default. Reads the frozen CSVs. Fully
  populated, no hardware needed.
- **Serial (ESP32)** — only usable when a real ESP32 is on a COM port. Port
  candidates are filtered to ESP32 vendor IDs, Bluetooth COM ports are
  rejected, and if no device passes the filter the app shows no status cards
  rather than fabricating placeholder readings.

Filters: date range, recent-N (capped at 500 rows), anomaly-only,
sensor-suspect-only, and CSV download of the filtered view.

---

## Data contract across the boundary

`data/control_output.csv` is the seam where phases 4 and 5 meet. One row per
input row, in order, 2016 rows, no missing values, 13 columns:

```
timestamp, predicted_occupancy, actual_occupancy, temperature, light_level,
light_status, hvac_status, anomaly_detected, anomaly_reason,
temperature_sensor_health, light_sensor_health, motion_sensor_health,
control_reason
```

Two design points in that schema carry real information:

- **`hvac_status` is one three-valued field**, not two booleans. `COOLING` and
  `HEATING` are mutually exclusive *by representation*, so "both on" is not a
  state the schema can hold. The thresholds are also asserted mutually
  exclusive at import time.
- **`control_reason` and `anomaly_reason` are stored per row.** Every actuator
  decision is traceable to the exact rule that produced it, so a reviewer can
  check a single row by hand instead of trusting an aggregate.

## Safety invariant

**A `suspect` sensor never drives its actuator.** If a sensor is flagged suspect,
its dependent actuator is forced `OFF` and the reason records which sensor was
distrusted. The reading is never replaced with a plausible-looking number — an
untrusted reading is treated as *insufficient evidence to act*, not as evidence
for the opposite action.

This is asserted across all 2016 replay rows by
`tests/test_phase5_replay.py::test_fail_safe_never_acts_on_a_suspect_sensor`,
and a companion test fails if the invariant ever passes vacuously.
