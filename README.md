# Smart Occupancy IoT + ML System

College IoT + Machine Learning capstone:
**Occupancy-Based Lighting/HVAC**

Two layers — the **IoT firmware layer** (Phase 2B) and the **ML/analysis layer**
(Phases 2C–5). The ML pipeline, anomaly rules and control logic are complete and
verified; live hardware validation is still outstanding.

---

## 1. Layers and phase scope

### Layer 1 — real hardware / firmware (Phase 2B)

```
ESP32 DevKit C V4
   ├── PIR motion sensor  (GPIO 27)
   ├── LDR photoresistor  (GPIO 34, ANALOG)
   └── DS18B20            (GPIO 4, 1-Wire)
            ↓
      Sensor readings
            ↓
   Basic control logic (hardware sanity check)
            ↓
   Serial output @ 115200 baud
            ↓
   LOCAL COMPILATION + WOKWI
```

### Layer 2 — historical dataset and ML pipeline (Phases 2C–5)

```
Python simulation (src/data_generator.py)
            ↓
   data/sensor_data.csv   7 days @ 5 min = 2016 rows   (FROZEN)
            ↓
   Phase 3: EDA + preprocessing
            ↓
   Phase 4: occupancy / temperature models, rolling-Z anomaly baseline
            ↓
   Phase 5: flatline detection, sensor health, fail-safe control
            ↓
   data/control_output.csv   per-interval decisions
```

**Status at a glance**

| Phase | Deliverable | State |
| --- | --- | --- |
| 2B-FIX | Firmware builds, Wokwi dependencies declared | done, no live hardware |
| 2C | Reproducible 2016-row dataset | done, frozen by hash |
| 3 | EDA, figures, leakage-aware splits | done |
| 4 | Models, evaluation, ablations, persisted | done |
| 5 | Detection rules, sensor health, control, replay, tests | done |
| 6 | Streamlit dashboard over the frozen outputs | done (75 tests pass) |
| 7 | Architecture diagram, report, viva, demo, limitations, audit | done (44/44 checks) |

**Why a Python simulation instead of harvesting Wokwi?** Wokwi demonstrates
the hardware layer. The firmware emits one sample per 5 s, so collecting 2016
samples would take roughly 2.8 days of real time. The ML pipeline is therefore
trained on this simulated history. Both layers share the same thresholds and
the same normalised scales, so they stay consistent.

**Not implemented yet:** database, cloud, MQTT, on-device inference. The ML
models and the Z-score detector are implemented (Phases 4–5); the Python serial
ingestion exists in the dashboard (Phase 6) but has not been run against real
hardware.

---

## 2. Project layout

```
.
├── app.py                   Streamlit dashboard (Phase 6)
├── platformio.ini          Local build configuration
├── libraries.txt           Wokwi Arduino library list (Phase 2B-FIX)
├── requirements.txt        Pinned Python dependencies
├── wokwi.toml              Wokwi CLI config, points at the PlatformIO build
├── src/
│   ├── main.cpp            CANONICAL firmware source
│   ├── config.py           Dataset + control configuration (Phase 2C)
│   ├── data_generator.py   Historical dataset generator (Phase 2C)
│   ├── preprocessing.py    Load / validate / split / features (Phase 3)
│   ├── eda.py              EDA report + figures (Phase 3)
│   ├── modeling.py         Models, evaluation, ablations, persistence (Phase 4)
│   ├── detection.py        Z-score + flatline rules, sensor health (Phase 5)
│   ├── control_logic.py    Pure fail-safe control rules (Phase 5)
│   ├── phase5_run.py       Historical replay driver (Phase 5)
│   ├── dashboard_data.py   Frozen-data access for the dashboard (Phase 6)
│   ├── architecture_diagram.py  Renders docs/architecture.png (Phase 7)
│   ├── final_audit.py      44-check functional/regression audit (Phase 7)
│   ├── export_appendix.py  Full code listing generator (Phase 7)
│   └── export_report_docx.py  Report -> .docx converter (Phase 7)
├── models/
│   ├── occupancy_model.pkl     DecisionTreeClassifier (Phase 4)
│   └── temperature_model.pkl   LinearRegression (Phase 4)
├── data/
│   ├── sensor_data.csv     Generated dataset, 2016 rows (FROZEN)
│   └── control_output.csv  Replay: detection + control decisions (Phase 5)
├── tests/
│   ├── test_control_logic.py   Control truth table + flatline units
│   ├── test_phase5_replay.py   Schema, safety invariants, determinism
│   └── test_dashboard.py       Phase 6 acceptance criteria
├── visualizations/
│   └── raw_sensor_overview.png
├── sketch.ino              Wokwi copy of main.cpp (kept byte-identical)
├── diagram.json            Wokwi circuit (original, unchanged)
├── wokwi/
│   ├── diagram.json        Same circuit, for PlatformIO tooling
│   └── libraries.txt
├── docs/
│   ├── PROJECT_REPORT.md   Full capstone report (Phase 7)
│   ├── PROJECT_REPORT.docx Submission-ready Word version (Phase 7)
│   ├── ARCHITECTURE.md     5-layer architecture, layer by layer (Phase 7)
│   ├── architecture.png     Labeled architecture diagram (Phase 7)
│   ├── VIVA.md             Full-pipeline explanation + Q&A (Phase 7)
│   ├── DEMO_PROCEDURE.md   10-12 minute demo script (Phase 7)
│   ├── LIMITATIONS.md      Every limitation, with measured magnitude (Phase 7)
│   ├── APPENDIX_CODE.md    Generated full code listing (Phase 7)
│   ├── PHASE 1 - Project Definition & Setup.md
│   └── Project Context - Smart Occupancy-Based Lighting & HVAC.md
└── wokwi-project.txt       Link to the live Wokwi project
```

### Why there are two copies of the firmware

`src/main.cpp` and `sketch.ino` contain **identical code**. They are not two
implementations — they are the same implementation expressed in the two
directory layouts the two build tools require:

| Tool | Required filename |
| --- | --- |
| PlatformIO | `src/main.cpp` |
| Wokwi.com | `sketch.ino` next to `diagram.json` |

**Editing rule:** make the change in `src/main.cpp`, then re-sync:

```powershell
Copy-Item src\main.cpp sketch.ino -Force
```

Verify the copies match:

```powershell
(Get-FileHash src\main.cpp).Hash -eq (Get-FileHash sketch.ino).Hash   # must be True
```

---

## 3. Hardware pin mapping

| Component | Signal | ESP32 pin | Notes |
| --- | --- | --- | --- |
| PIR motion sensor | VCC | 3V3 | |
| | GND | GND.1 | |
| | OUT | **GPIO 27** | `0` = no motion, `1` = motion |
| LDR / photoresistor | VCC | 3V3 | |
| | GND | GND.1 | |
| | **AO (analog)** | **GPIO 34** | ADC1_CH6, input-only, 0–4095 |
| | DO | *not used* | Digital threshold output deliberately ignored |
| DS18B20 | VCC | 3V3 | |
| | GND | GND.1 | |
| | DQ | **GPIO 4** | 1-Wire data |
| | DQ pull-up | 4.7 kΩ → 3V3 | via `r1` |
| Lighting LED | anode | **GPIO 18** | via 220 Ω (`r2`) |
| | cathode | GND.2 | |
| HVAC LED | anode | **GPIO 19** | via 220 Ω (`r3`) |
| | cathode | GND.2 | |

These pins match `wokwi/diagram.json` exactly. The circuit was **not** modified.

---

## 4. Build (local)

PlatformIO Core is required:

```powershell
python -m pip install --upgrade platformio
```

Compile:

```powershell
python -m platformio run
```

Expected result:

```
RAM:   [=         ]   6.6% (used 21688 bytes from 327680 bytes)
Flash: [==        ]  21.5% (used 281293 bytes from 1310720 bytes)
========================= [SUCCESS] Took 58.33 seconds =========================
```

Serial monitor (needs a real ESP32 connected over USB):

```powershell
python -m platformio device monitor
```

No ESP32 is currently attached to this machine — `pio device list` reports only
Bluetooth COM3/COM4 — so no upload has been performed and no live serial
capture is included in this report.

### Python checks

```powershell
python -m pip install -r requirements.txt   # first time only
python -m pytest tests -q                   # 39 tests
```

Expected: `39 passed`. The suite covers the control truth table, the flatline
rules, the replay schema, the fail-safe invariant across all 2016 rows, and
reproducibility of the replay output. It also asserts that the frozen dataset
hash has not changed, so a Phase 5 regression cannot pass by quietly rewriting
the source data.

The Phase 5 replay itself is run with:

```powershell
python src/phase5_run.py
```

This rewrites `data/control_output.csv` and prints the full metrics report. It
loads the persisted models and never retrains, so re-running it is safe and
must leave the output byte-identical — which one of the tests asserts.

---

## 5. Dependencies

Declared in `platformio.ini`, resolved automatically by the build:

| Library | Version | Purpose |
| --- | --- | --- |
| `paulstoffregen/OneWire` | ^2.3.8 (resolved 2.3.8) | 1-Wire bus protocol |
| `milesburton/DallasTemperature` | ^3.11.0 (resolved 3.11.0) | DS18B20 driver |

Toolchain pulled in by `platform = espressif32`:

| Package | Version |
| --- | --- |
| framework-arduinoespressif32 | 4.20017.260907 |
| toolchain-xtensa-esp32 | 8.4.0+2021r2-patch5 |
| tool-esptoolpy | 4.11.0 |

Board: `esp32dev` (ESP32 DevKit C V4 — matches the Wokwi board part
`board-esp32-devkit-c-v4`).

> **Wokwi note:** Wokwi provides OneWire and DallasTemperature automatically.
> `"dependencies": {}` in `diagram.json` is intentional and left unchanged.

---

## 6. Configuration constants

Control thresholds are defined in **two places that must stay in sync**:
`src/main.cpp` (firmware) and `src/config.py` (dataset). They are the same
values, so the simulated history and the real firmware describe one room.

| Constant | Value | Lives in | Meaning |
| --- | --- | --- | --- |
| `LIGHT_THRESHOLD` | 300 | both | Below this normalised light level, switch lighting on |
| `HVAC_COOLING_THRESHOLD` | 26.0 °C | both | Above this, switch HVAC on |
| `HVAC_HEATING_THRESHOLD` | 20.0 °C | `config.py` | Reserved for a later phase, not used yet |
| `SENSOR_INTERVAL` | 5000 ms | `main.cpp` | Firmware sampling period |
| `ADC_RAW_MAX` | 4095 | `main.cpp` | ESP32 12-bit ADC full scale |
| `LIGHT_SCALE_MAX` | 1000 | `main.cpp` | Normalised light scale full scale |
| `TEMP_PLAUSIBLE_MIN/MAX` | −55 / 125 °C | `main.cpp` | Physical plausibility band |
| `TEMP_MIN` / `TEMP_MAX` | 18.0 / 32.0 °C | `config.py` | Plausible band for normal samples |
| `SEED` | 42 | `config.py` | Makes the dataset reproducible |
| `DATASET_DAYS` | 7 | `config.py` | Length of generated history |
| `SAMPLING_INTERVAL_MINUTES` | 5 | `config.py` | Matches the firmware interval |
| `ANOMALY_RATE` | 0.02 | `config.py` | Injected anomaly fraction (spec target 1–3%) |

> `src/` is shared between the C++ firmware and the Python analysis code.
> `platformio.ini` sets `build_src_filter = +<main.cpp>` so only the
> firmware is ever handed to the ESP32 compiler. Do not remove that line.

---

## 7. Control logic (hardware sanity check)

Occupancy is the **raw PIR motion signal** in this phase. The ML occupancy
model is not wired in yet.

```text
LIGHT_ON  = (motion == 1) AND (light_level < 300)
HVAC_ON   = (motion == 1) AND (temp_valid) AND (temperature > 26.0)
```

Boundary behaviour:

| motion | light_level | temperature | valid | light | hvac |
| --- | --- | --- | --- | --- | --- |
| 0 | 100 | 24.00 | yes | 0 | 0 |
| 1 | 299 | 24.00 | yes | 1 | 0 |
| 1 | 300 | 24.00 | yes | 0 | 0 |
| 1 | 301 | 24.00 | yes | 0 | 0 |
| 1 | 100 | 26.00 | yes | 1 | 0 |
| 1 | 100 | 26.01 | yes | 1 | 1 |
| 0 | 100 | 30.00 | yes | 0 | 0 |
| 1 | 100 | −127.00 | **no** | 1 | 0 |

---

## 8. Serial output protocol

Baud rate **115200**. Configured as `monitor_speed` in `platformio.ini` and
matched by `Serial.begin(115200)` in `setup()`.

### Startup banner

```text
======================================
 Smart Occupancy IoT System
 ESP32 + PIR + LDR + DS18B20
======================================
[INIT] DS18B20 found on GPIO 4, devices=1
light_level is a NORMALISED 0-1000 scale, not calibrated lux.
...
timestamp,motion,light_level,temperature,light_status,hvac_status
```

### Data records — one every 5 seconds

```text
DATA,timestamp,motion,light_level,temperature,light_status,hvac_status
```

Example:

```text
DATA,5000,0,742,24.31,0,0
DATA,10000,1,218,24.38,1,0
DATA,15000,1,185,24.44,1,0
DATA,20000,1,192,26.47,1,1
```

| Field | Type | Meaning |
| --- | --- | --- |
| `timestamp` | int | Elapsed milliseconds since boot (`millis()`) |
| `motion` | 0/1 | PIR state |
| `light_level` | 0–1000 | **Normalised** light scale (not calibrated lux) |
| `temperature` | float, 2 dp | °C, or `-127.00` if the reading is invalid |
| `light_status` | 0/1 | Lighting LED |
| `hvac_status` | 0/1 | HVAC LED |

### Error records

Emitted **in addition to** the corresponding `DATA` line, so a record is never
silently accepted as valid:

```text
ERR,TEMP_INVALID,raw=-127.00,action=HVAC_FORCED_OFF
```

An invalid temperature can never switch HVAC on (fail-safe).

### Honest scope note on the light scale

`light_level` is `map(analogRead(34), 0, 4095, 0, 1000)`, clamped to 0–1000.
It is a **normalised ADC scale, not calibrated physical lux**. A real lux
conversion needs a calibrated sensor transfer curve and ambient reference,
neither of which is in scope here. Do not describe this value as lux in the
project report.

---

## 9. Known limitations

1. **No live hardware run yet.** Compilation is verified and the circuit passes
   `wokwi-cli lint`, but the firmware has not been executed on a physical ESP32
   or inside the Wokwi simulator. See section 9.1.
2. **`light_level` is not calibrated lux** (see section 8). The same normalised
   0–1000 scale is used by the firmware and the generated dataset.
3. **Occupancy is raw PIR motion** in the firmware. It is a sampled ground-truth
   label in the dataset, not an ML prediction.
4. **Two firmware copies exist** (`src/main.cpp` + `sketch.ino`) and must be kept
   in sync manually per section 2.
5. **DS18B20 conversion takes ~750 ms** at the default 12-bit resolution. This is
   well inside the 5 s interval, so it is not currently a problem.
6. **A Z-score detector cannot find stuck-sensor anomalies.** In the generated
   dataset the frozen runs have global z-scores of only ~0.7 and ~1.2, so a
   threshold of 2 or 3 will miss all 12 stuck rows. Only spikes and drops are
   detectable by z-score. **Resolved in Phase 5** by a range-based flatline rule,
   which now flags both injected stuck runs in full (5/5 and 7/7). See section 13.
7. **Two hours of the day are single-class.** Hours 2 and 4 (early morning) are
   unoccupied in all 84 samples each, so they carry no occupancy signal. This
   is realistic, but a model trained on `hour` alone will learn nothing from
   those two buckets. The anti-leakage check requires at least 80% of hours to
   contain both classes; the current dataset is at 22/24 (91.7%).
8. **The Phase 3 test period is entirely weekend.** See section 12. This is the
   largest single risk to Phase 4 numbers.
9. **15 test rows (2.6%) have an exact feature twin in the training set.** They
   arise where the room is dark, motion-free and temperature repeats to two
   decimals in the small hours. All 14 twin groups carry the *same* occupancy
   label, so memorising them reproduces what the features already imply and the
   score is not meaningfully inflated. Asserting "zero duplicate features"
   would have failed on harmless data while measuring a real contradiction just
   as loosely.
10. **The Phase 5 flatline thresholds are tuned to this generator's noise.**
    The simulated stuck sensor freezes a value and then dithers by a last
    significant bit; real hardware drifts smoothly instead. Both the range
    thresholds and the measured false-positive counts would need re-deriving on
    real data, and the temperature rule in particular sits on an ambiguous
    boundary (the fault's range and a still night's range are both exactly
    0.03 °C, so 7 false positives in 2016 rows are the price of detecting it at
    all). See section 13.
11. **The heating branch is never exercised by the replay.** It is unreachable
    in this dataset because the generator makes cold coincide with absence — no
    row is both below 20 °C and predicted occupied. The rule is covered by unit
    tests only, and the replay prints branch reachability so this stays visible
    rather than looking like a working feature.

### 9.1 Wokwi validation status

`wokwi-cli lint wokwi/diagram.json` passes with **exit code 0** and no errors,
confirming all 9 parts and 19 connections are valid. Two informational notes are
raised for `board-esp32-devkit-c-v4` and `wokwi-ds18b20` being "undocumented";
both are documented upstream, so this is a stale part database in the CLI, not a
problem with the circuit.

Running the **simulation** is not possible from this machine: `wokwi-cli` requires
a Wokwi account token (`WOKWI_CLI_TOKEN`) to reach the build/simulation service.
Obtain one at <https://wokwi.com/dashboard/ci>, then:

```powershell
$env:WOKWI_CLI_TOKEN = "<your token>"
wokwi-cli --timeout 30000 --expect-text "DATA," .
```

---

## 10. Reproducing the dataset

```powershell
pip install -r requirements.txt
python src\data_generator.py
```

Regenerating always produces a byte-identical CSV, because `SEED = 42` in
`src/config.py`. The script prints measured statistics and runs 18 validation
checks, exiting non-zero if any fails.

To change the shape of the data, edit `src/config.py` only — not the generator.
Changing `SEED`, `DATASET_DAYS` or `ANOMALY_RATE` and re-running is the intended
way to explore alternatives.

The dataset schema is fixed at exactly these seven columns:

```text
timestamp, hour, motion, light_level, temperature, occupancy, is_anomaly
```

`is_anomaly` is **ground truth recorded at injection time**. It is not produced
by any statistical detector and must not be recomputed from the data.

---

## 11. Phase 3 — EDA and preprocessing

### Files

| File | Role |
| --- | --- |
| `src/preprocessing.py` | Reusable library: load, validate, split, features, leakage checks |
| `src/eda.py` | Phase 3 driver: prints the report, writes the 8 figures |

```powershell
python src\eda.py
```

`src/preprocessing.py` never writes to disk. `data/sensor_data.csv` is read-only
and is verified byte-identical after every Phase 3 run.

### Figures

| File | Shows |
| --- | --- |
| `sensor_time_series.png` | All 3 sensors over 7 days, stacked subplots (scales differ by 1000x) |
| `occupancy_distribution.png` | Occupied vs unoccupied counts |
| `motion_vs_occupancy.png` | Contingency matrix — related but not identical |
| `light_distribution.png` | Light histogram with the 300 control threshold |
| `temperature_time_series.png` | Temperature over time, anomalies marked, 26 °C line |
| `temperature_by_hour.png` | Box plot per hour — shows the daily cycle |
| `occupancy_by_hour.png` | Occupancy rate per hour; **hours 2 and 4 flagged red** |
| `anomaly_overview.png` | The 40 anomalous rows against normal |

### Measured temporal structure

| Series | lag-1 (5 min) | lag-12 (1 h) | lag-288 (1 day) |
| --- | --- | --- | --- |
| `temperature` | **0.9106** | 0.8694 | 0.9030 |
| `light_level` | 0.9475 | 0.8976 | 0.9455 |

> **Correction to the Phase 2C report.** It quoted temperature lag-1
> autocorrelation ≈ 0.999. That was measured on the smooth series *before*
> anomaly injection. The shipped CSV contains 40 injected anomalies (spikes,
> drops, frozen runs) which break local smoothness, so the real value on the
> file the model trains on is **0.911**. Still far too high for a random split,
> but 0.999 must not be quoted as a property of this data.
>
> Autocorrelation does not decay monotonically: lag-288 (0.903) exceeds lag-12
> (0.869) because the series is dominated by a daily cycle — "same time
> yesterday" is more similar than "one hour ago".

### Chronological split

| | Rows | Period | Occupancy rate |
| --- | --- | --- | --- |
| Train | 1440 | 2026-01-05 00:00 → 2026-01-09 23:55 | 52.36% |
| Test | 576 | 2026-01-10 00:00 → 2026-01-11 23:55 | 26.39% |

`train_test_split` is never imported. The boundary is a single timestamp, so
`max(train) < min(test)` holds by construction.

### Feature definitions

| Task | Features `X` | Target `y` |
| --- | --- | --- |
| Occupancy | `hour`, `motion`, `light_level`, `temperature` | `occupancy` |
| Temperature | `hour`, `motion`, `light_level`, `occupancy` | `temperature` |
| Anomaly | `motion`, `light_level`, `temperature` | `is_anomaly` |

`timestamp` and `is_anomaly` are rejected as inputs by an **import-time guard**
(`_assert_valid_features`), not by convention — a bad feature list raises at
import rather than quietly producing a great-looking score.

No scaler is applied. A decision tree is scale-invariant, and linear regression
on `hour`/`light_level`/`temperature` is well conditioned without one.

---

## 12. Phase 4 — models, evaluation, persistence

```powershell
python src\modeling.py
```

Trains all three components and evaluates each on **two** splits, then writes
`models/occupancy_model.pkl` and `models/temperature_model.pkl`.

| File | Role |
| --- | --- |
| `src/modeling.py` | `RollingZScore` detector, tuning, evaluation, ablations, persistence |

### Two evaluation splits

| Split | Train | Test | Purpose |
| --- | --- | --- | --- |
| **primary** (chronological) | 1440 rows, Mon–Fri | 576 rows, Sat–Sun | Headline. Deployment-realistic |
| `weekday_holdout` (diagnostic) | 1728 rows, incl. both weekends | 288 rows, Wednesday | Separates distribution shift from model weakness |

`weekday_holdout` is **not** chronologically pure — its training data contains
days *after* the holdout. That is exactly why it is a diagnostic and the
chronological split stays primary.

### Occupancy — Decision Tree

The unbounded default tree overfits badly (depth 20, 247 leaves, **99.86%**
train accuracy vs 72.4% test). Depth and leaf size are chosen by
`TimeSeriesSplit(5)` **on the training split only** → `max_depth=3`,
`min_samples_leaf=20`.

| Model / split | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| tuned, all sensors / primary | 0.7778 | 0.5451 | 0.9539 | 0.6938 |
| tuned, all sensors / weekday_holdout | 0.8472 | 0.9044 | 0.7987 | 0.8483 |
| default (overfit) / primary | 0.7240 | 0.4867 | 0.8421 | 0.6169 |
| hour only / primary | 0.7222 | 0.4872 | 1.0000 | 0.6552 |
| hour only / weekday_holdout | 0.8681 | 0.9394 | 0.8052 | 0.8671 |

**Read this against the trivial baseline.** On the primary split, "always
predict unoccupied" scores **0.7361** accuracy — the tree manages only
**+0.042** over it. On `weekday_holdout` the same model is far clear of
baseline (0.8472 vs 0.5347). The primary number is depressed by the weekend
shift, not by a weak model.

### Temperature — Linear Regression

No `temperature_lag1` feature, despite 0.9106 lag-1 autocorrelation, because it
would turn prediction into interpolation and inflate R².

| Split | MAE | RMSE | R² |
| --- | --- | --- | --- |
| primary | 1.3404 | 1.9532 | 0.6715 |
| weekday_holdout | 1.3165 | 1.6541 | 0.7606 |

### Anomaly — rolling Z-score

Rolling (not global) z-score on `motion`, `light_level`, `temperature`. The
baseline for each sample is the previous `window` samples only, and a **full**
window is required before judging. `window` and `k` are grid-searched on the
training split only → `window=288` (1 day), `k=2.0`.

| Split | Precision | Recall | F1 |
| --- | --- | --- | --- |
| primary | 1.0000 | 1.0000 | 1.0000 |
| weekday_holdout | 0.5000 | 0.1111 | 0.1818 |

> **Do not quote F1 = 1.000.** The primary test period happens to contain only
> 7 anomalies, all spikes or drops, which any z-score catches. The honest
> figure is **dataset-wide: 18 of 33 anomalies detected (54.5%)**, 3 false
> positives in 1728 rows. Stuck sensors are undetectable by this method — a
> frozen value makes the rolling deviation zero once the window fills.

### Ablations

| Ablation | Result |
| --- | --- |
| hour only vs all sensors (F1) | primary **+0.0386** for all sensors, weekday_holdout **−0.0189** |
| twins excluded from primary test | accuracy −0.0059, precision/recall/F1 unchanged |

The hour-only comparison **does not** support a clean "more sensors is better"
claim: the effect is small and flips sign between splits. The tuned tree gives
`motion` zero importance and leans on `temperature` (0.75) and `light_level`
(0.22) — largely because occupancy, motion and temperature are all driven by the
same underlying presence in this simulation.

---

## 13. Phase 5 — detection, sensor health, and control logic

### Files

| File | Role |
| --- | --- |
| `src/detection.py` | Phase 4 z-score (imported, unchanged) + flatline rules, reasons, sensor health |
| `src/control_logic.py` | Pure rule-based control with fail-safe behaviour. No I/O, no model |
| `src/phase5_run.py` | Historical replay driver; writes `data/control_output.csv` |
| `data/control_output.csv` | 2016 rows of detection + control decisions |
| `tests/test_control_logic.py` | 26 tests: full control truth table + flatline unit tests |
| `tests/test_phase5_replay.py` | 13 tests: schema, safety invariants, determinism |

Reproduce with `python src/phase5_run.py`; verify with `python -m pytest tests -q`
(39 passed).

### Why the z-score alone was not enough

Phase 4 measured the inherited detector at **18 of 33 anomalies dataset-wide
(54.5%)**, missing every stuck sensor. The reason is structural: a frozen value
makes both the local standard deviation *and* the deviation from the baseline
approach zero, so the ratio that a z-score tests never grows. No choice of
`window` or `k` fixes that.

### The flatline rule is on range, not on repeated values

The obvious rule — "flag N identical readings" — does not work on this data, in
either direction. Measured on the non-anomalous rows, natural identical-value
runs reach 3 samples for temperature, 9 for light and 49 for motion, so a short
exact-equality rule fires constantly. But the *injected* faults are not exact
copies either; they dither by a bit:

```
temperature  [26.75, 26.72, 26.73, 26.73, 26.73]   range 0.03 C
light_level [573, 574, 572, 574, 572, 574, 573]   range 2 lux
```

An exact-equality rule therefore detects **none** of them. Detection is on the
rolling **range** (max − min) across a trailing window, with a duration.

### Thresholds, and the honest limit of a range-only rule

| Sensor | Window | Max range | Extra condition | Injected fault caught | False positives |
| --- | --- | --- | --- | --- | --- |
| temperature | 6 (30 min) | 0.03 °C | — | 5 / 5 | 7 |
| light_level | 7 (35 min) | 2 lux | mean ≥ 300 | 7 / 7 | 1 |
| motion | 288 (24 h) | 0 | motion=1 or light ≥ 500 | n/a | 0 |

Two of these deserve to be stated plainly rather than buried:

- **Temperature cannot be separated from a quiet night.** The fault's range is
  exactly 0.03 °C and the overnight natural minimum is also 0.03 °C. Below
  0.03 the fault is invisible; at 0.03 a few genuinely still night windows are
  also reported. The rule accepts 7 false-positive rows out of 2016 (0.35%) to
  detect the fault at all. A 5-sample window also works but doubles the false
  positives (12 → 7 at 6 samples), so 6 is the measured operating point.
- **The light mean-gate is the single most important part of the light rule.**
  A constant reading of `0` is real darkness, not a fault. Without requiring a
  lit-level mean, the rule fires **18** times on ordinary nights instead of 1.

`sensitivity_table()` prints the full sweep, so the operating point is
checkable rather than taken on trust.

### Sensor health vs environmental anomaly

These are deliberately different signals, because the two call for different
responses:

- A **flatline** is a signature no real environment reproduces → sensor
  `suspect` → the fail-safe may act on it.
- A **z-score hit** is a real hot/cold/bright period → `normal` health, still
  reported as an anomaly. A temperature spike is not a broken thermometer.

Only flatlines can mark a sensor suspect. Reasons are prioritised
specific-before-generic, and the underlying flags stay in the result so the
overlap between a flatline and a z-score hit is auditable rather than lost.

### Detection results

Scored on rows 288–2015; the 288 warmup rows are excluded because the inherited
z-score cannot judge them. Dataset-wide, using the primary split's test period
boundaries as in Phase 4:

| Detector | TP | FP | FN | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| rolling z-score (Phase 4, inherited) | 18 | 3 | 15 | 0.857 | 0.545 | 0.667 |
| z-score + flatlines (Phase 5) | 25 | 10 | 8 | 0.714 | 0.758 | 0.735 |

Phase 5 trades precision for recall: both stuck runs go from 0 % to 100 %
detected, at the cost of 7 extra false positives. F1 improves 0.667 → 0.735.
All 8 remaining misses in the scored region are single-sample events — 3
`light_drop`, 4 `motion_false_trigger`, 1 `temperature_spike`. No flatline is
missed: all 12 injected stuck-sensor rows (5 temperature at rows 228–232,
7 light at rows 748–754) are flagged. (Dataset-wide there are 10 misses, 2 of
which fall inside the 288-row warmup and are excluded from this table. Scored
over all 2016 rows instead, the detector is TP 30 / FP 11 / FN 10, precision
0.732, recall 0.750, F1 0.741.)

### Control logic and fail-safe behaviour

Pure functions of (predicted occupancy, temperature, light level, sensor
health), so the truth table is asserted exactly rather than statistically.

- **Light** `ON` iff predicted occupancy is 1, light sensor healthy, and
  `light_level < 300`. Strict `<`, so exactly 300 counts as already lit.
- **HVAC** is a single three-valued field: `COOLING` above 26 °C, `HEATING`
  below 20 °C, else `OFF`. One field rather than two booleans, so both-on is
  not representable; the thresholds are also asserted mutually exclusive at
  import time.
- **Fail-safe:** a `suspect` required sensor forces its actuator `OFF`, and the
  reason records which sensor was distrusted. The reading is never replaced with
  a plausible-looking number. An untrusted reading is treated as *insufficient
  evidence to act*, not as evidence for the opposite action.

The fail-safe invariant — *a suspect sensor never drives its actuator* — is
asserted across all 2016 replay rows, not only in the unit tests, and a
companion test fails if the invariant ever passes vacuously.

### Historical replay

`data/control_output.csv`, 2016 rows, no missing values:

| Field | Range observed |
| --- | --- |
| `light_status` | OFF 1732, ON 284 (14.09 %) |
| `hvac_status` | OFF 1307, COOLING 709 (35.17 %), **HEATING 0** |
| `temperature_sensor_health` | suspect 12 (0.60 %) |
| `light_sensor_health` | suspect 8 (0.40 %) |
| `motion_sensor_health` | suspect 0 |
| fail-safe overrides | light 0, HVAC 7 intervals |

**Heating never fires, and that is a property of the data, not a bug.** The
generator makes cold coincide with absence: 324 rows are below 20 °C, but
**none** of them has predicted occupancy 1 (the coldest predicted-occupied row is
22.47 °C). The replay prints the gating counts for every branch — including the
empty ones — precisely so this is distinguishable from a broken rule. The
heating rule itself is covered by unit tests.

The light fail-safe override count is 0 because the light fault sits at 573 lux,
which is above the 300 threshold, so a healthy-sensor rule would have left the
light off anyway. The fail-safe path is genuinely exercised for HVAC (7
intervals) and is asserted by test either way.

**No energy or cost saving is reported.** Nothing in this project models
actuator power draw, so any savings figure would be invented. What is countable
honestly is how often each actuator is on, and how often the fail-safe
overrode the healthy-sensor decision.

### Occupancy prediction in the replay (context only)

Applying the persisted Phase 4 model to all 2016 rows gives accuracy 0.8408,
precision 0.7887, recall 0.8819, F1 0.8327. This is a **different quantity** from
the Phase 4 primary-split score and is not a revision of it — only the test
split is a generalisation estimate, and 1440 of these rows were used to train
the model.

### What is frozen

`sensor_data.csv`, both `.pkl` models, `modeling.py` and `preprocessing.py` are
byte-identical to their Phase 4 hashes, asserted by
`test_source_dataset_is_unchanged`. The firmware builds unchanged (RAM 6.6 %,
Flash 21.5 %), and `src/main.cpp` and `sketch.ino` remain identical copies. No
retraining happens anywhere in this phase; the model is loaded from disk.

### Still outstanding

1. **Live hardware is untested.** No ESP32 has been attached and Wokwi cloud
   simulation needs `WOKWI_CLI_TOKEN`, so the firmware has never seen real sensor
   values, and the control logic has never driven a real relay.
2. **The thresholds are tuned to this generator's noise model.** Real hardware
   drifts rather than freezing, so the range thresholds and the measured
   false-positive counts would both need re-deriving on real data.
3. **Occupancy remains weak on the primary split** (+0.042 over a trivial
   baseline). Phase 6 presents that number with the baseline beside it, as
   promised above.

---

## 14. Phase 6 — Streamlit dashboard

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Launches on `http://localhost:8501`; the health endpoint returns `200 ok`.
Verified: `python -m streamlit run app.py --server.headless true` plus a
headless `AppTest` render, both with zero exceptions.

| File | Role |
| --- | --- |
| `app.py` | UI only — layout, charts, filters. Contains no training code. |
| `src/dashboard_data.py` | Frozen-data access. Deliberately **free of any Streamlit import** so it is testable from plain pytest. |
| `tests/test_dashboard.py` | 36 tests mapping to the Phase 6 acceptance criteria |

### The dashboard does not pretend to have hardware

This is the governing constraint of Phase 6, and it is enforced in three places
rather than mentioned once in a comment:

- The sidebar has an explicit **mode selector**. `Historical Simulation` is the
  default and renders the frozen CSVs.
- `ESP32 Serial` mode is offered, but the port scan **rejects anything that is
  not an ESP32**. On this machine both COM3 and COM4 are Bluetooth SPP links, so
  they are listed with the reason they were rejected rather than offered as a
  data source. A live panel filled with simulated readings is the specific
  failure this design prevents.
- In serial mode with no device present, the status cards are **not rendered at
  all**. A test asserts that no "Occupied" metric appears, so the page cannot
  quietly fall back to historical values.

The simulation banner appears above every view:

> **Simulation Mode — historical sensor data** — Every value on this page comes
> from the frozen files `data/sensor_data.csv` and `data/control_output.csv`. No
> ESP32 is streaming data, and nothing here is live.

### 1. System status

Occupancy, temperature, light level, lighting, HVAC, sensor health, anomaly and
control reason for the most recent observation. In simulation mode the heading
carries the record's own timestamp and the caption states plainly that it is a
historical record, not a live reading.

### 2. ML performance

Occupancy accuracy / precision / recall / F1 and temperature MAE / RMSE / R²,
**recomputed from the persisted models on the frozen splits** — inference only.
A consistency check runs on every page load and confirms the recomputed primary
metrics equal the Phase 4 report; the sidebar shows a green success or a red
failure.

**The weekday diagnostic is labelled as a different quantity, deliberately.**
Only the primary models were persisted in Phase 4. The Phase 4 weekday figure
(F1 0.8483) came from a model trained separately on the weekday training split,
which was never saved — so the dashboard cannot reproduce it from disk. Rather
than retrain to match, it shows the *persisted primary model* scored on the
Wednesday holdout (F1 **0.8618**) with an on-screen warning that the two numbers
are not comparable. Using one fixed model is what makes the panel useful: it
separates **distribution shift** from **model weakness**. The model scores
*better* on Wednesday than on the primary test period, so the weekend window is
the harder distribution — not the weekday result the better one.

The occupancy panel also shows the trivial baseline, computed from the data:

| Split | Always-unoccupied accuracy | Model accuracy | Gain |
| --- | --- | --- | --- |
| primary chronological test | 0.7361 | 0.7778 | **+0.0417** |
| weekday holdout | 0.4653 | 0.8542 | +0.3889 |

The primary test window is only 26.39 % occupied, which is what makes the
always-unoccupied baseline so strong there. This baseline is **computed, not
hardcoded** — a hardcoded copy is a trap here, because the training window is
52.36 % occupied and using that rate turns a genuine +0.042 gain into a
fictitious +0.3014. A test guards it.

### 3. Sensor and anomaly monitoring

Detection counts from the frozen Phase 5 output — 2016 rows, 40 injected
anomalies, 41 flagged, and the confusion matrix on rows 288+ (TP 25, FP 10, FN
8, precision 0.714, recall 0.758, F1 0.735), identical to the Phase 5 report.

Three Plotly charts, each with anomaly markers (`x`) and sensor-health events
(open orange circles) overlaid on the raw trace:

- Temperature over time, with the 26 °C / 20 °C HVAC thresholds marked.
- Light level over time, with the 300 lighting threshold marked.
- Motion against predicted and actual occupancy.

Health is raised by flatline signatures only, so a z-score hit leaves the sensor
`normal` — visible on the charts as an `x` without an accompanying health marker.

### 4. Intelligent control

Duty-cycle summary (lighting ON 14.09 %, HVAC active 35.17 %, fail-safe
violations 0) and a filterable table.

The full 2016 rows are **never rendered at once**. Filters are: date range,
most-recent-N (capped at 500), anomalies-only, and sensor-suspect-only. `N` is
applied *after* the other filters, so "recent" means recent within the
selection rather than recent within the file. The filtered view can be
downloaded as CSV for the rows the page does not show.

The panel reports duty cycle and fail-safe overrides, and states explicitly that
**no energy or cost saving is reported** — nothing in this project models
actuator power draw, so a savings figure would be invented.

### Phase 6 acceptance criteria

| Criterion | How it is verified |
| --- | --- |
| Streamlit launches successfully | `streamlit run` on :8512, `_stcore/health` → `200 ok`, empty stderr; `AppTest` render with 0 exceptions |
| No ML retraining inside `app.py` | AST walk of every call in `app.py` and `dashboard_data.py` against a forbidden-API set (`fit`, `train_test_split`, `GridSearchCV`, …) |
| Persisted models loaded, not retrained | Asserts `joblib.load(OCCUPANCY_MODEL)` and `joblib.load(TEMPERATURE_MODEL)` are used, and that `fit_occupancy` / `fit_temperature` / `tune_decision_tree` appear nowhere |
| Dashboard reads frozen datasets | Paths asserted in source; the dataset and both models are SHA-256 verified on every load |
| Historical controls match `control_output.csv` | 55 sampled rows re-derive their `light_status` / `hvac_status` from their own inputs via `control_logic` — 0 mismatches |
| Anomaly counts match Phase 5 | TP 25, FP 10, FN 8, precision/recall/F1 asserted to 3 dp; flatline counts 12/8/0 |
| Metrics match Phase 4 | Recomputed primary metrics asserted equal to the recorded Phase 4 values, including the full confusion matrix |
| Filters work | 9 tests: recent-N, ordering, cap, date range inclusive endpoints, anomaly-only, suspect-only, combination order |
| No fabricated live data | Port matcher unit-tested with fake ESP32 vs Bluetooth ports; asserts **zero** candidates on this machine; asserts no status card renders in serial mode |
| Source dataset SHA-256 unchanged | `2D38D741…D7D091` verified |
| All 39 existing tests remain green | Full suite now **75 passed** |
| Firmware still builds | `platformio run` → SUCCESS, RAM 6.6 %, Flash 21.5 % (unchanged) |
| Clear Simulation Mode label | Asserts the banner text, that the labels come from shared constants, and that simulation is the default radio option |

### Still outstanding

Live serial mode is implemented and gated, but **unexercised** — no ESP32 has
been connected, so the `DATA,` parser and the port matcher have only been
tested against recorded text and simulated ports. The firmware has still never
driven a real relay.

---

## 14. Phase 7 — integration, documentation and the final audit

Phase 7 adds no new model. Its job is to make the project deliverable and every
claim in it checkable.

### What was added

| Artefact | Purpose |
| --- | --- |
| `docs/architecture.png` | Labeled 5-layer diagram (sensor → connectivity → processing → AI model → application) |
| `docs/ARCHITECTURE.md` | Layer-by-layer description, pin map, data contract, safety invariant |
| `docs/PROJECT_REPORT.md` | The full report in the required format, 11 sections + references + appendix |
| `docs/PROJECT_REPORT.docx` | Submission-ready Word version |
| `docs/APPENDIX_CODE.md` | Generated full code listing (5786 lines across 21 files) |
| `docs/VIVA.md` | Full-pipeline explanation for **every** member, plus 20 hard Q&A |
| `docs/DEMO_PROCEDURE.md` | 10–12 minute demo script with fallback table |
| `docs/LIMITATIONS.md` | 14 limitations, each with a measured magnitude |
| `src/final_audit.py` | The 44-check audit described below |
| `src/architecture_diagram.py` | Regenerates the diagram |
| `src/export_appendix.py` | Regenerates the code listing |
| `src/export_report_docx.py` | Converts the report to `.docx` |

### The final audit

```bash
python src/final_audit.py
```

One command, **44 checks**, grouped as:

- **FUNCTIONAL (20)** — the required documents and diagram exist; the dashboard
  loads; the dataset, control output and models load; filters work; the control
  table is capped; the dashboard recomputes Phase 4 metrics by inference;
  serial mode offers no device without hardware.
- **REGRESSION (20)** — all **six** SHA-256-pinned artifacts unchanged
  (dataset, both models, `modeling.py`, `preprocessing.py`, and now
  `control_output.csv`); no training API in any shipped path; the Phase 4 fit
  helpers are never called; the persisted models load; the recomputed metrics
  equal the recorded Phase 4 values; the anomaly counts match Phase 5; the
  75-test suite is green; the firmware still builds; the firmware copies are
  byte-identical; and the fail-safe invariant holds across all 2016 rows.
- **WORDING (4)** — no *narrative* document claims a demonstrated energy saving,
  and the simulated dataset is disclosed in the README, the report and the
  limitations file.

The wording checks are the part worth noting: the audit reads the documentation
and fails if it finds a claim the evidence does not support. That is a
mechanical guard against the most common way a report like this becomes
dishonest. The generated code listing `docs/APPENDIX_CODE.md` is excluded from
that scan — it is a verbatim dump of every source file, so it necessarily
contains the audit's own banned-phrase list. The exclusion is printed in the
check name, so it is visible rather than silent.

### Reproducing the whole project from scratch

```bash
pip install -r requirements.txt
python src/phase5_run.py            # replay -> data/control_output.csv
python src/architecture_diagram.py  # -> docs/architecture.png
python src/export_appendix.py       # -> docs/APPENDIX_CODE.md
python src/export_report_docx.py    # -> docs/PROJECT_REPORT.docx
python -m pytest tests -q           # 75 passed
python src/final_audit.py           # 44/44
python -m platformio run            # SUCCESS
streamlit run app.py                # http://localhost:8501
```

### Two corrections Phase 7 made to earlier phases

Recording these because both were found by reading the code rather than trusting
the documentation:

1. **The anomaly miss breakdown in this README was wrong.** It claimed 4
   `light_drop` and 3 `motion_false_trigger` among the 8 scored misses. Measured
   from the data, the correct split is **3 `light_drop`, 4
   `motion_false_trigger`, 1 `temperature_spike`** (rows 449, 830, 1019, 1022,
   1072, 1187, 1252, 1364). The totals were right; the attribution was not.
2. **`src/config.py` carried a stale comment** claiming heating control was
   "NOT implemented yet". The Python control layer has implemented and
   unit-tested heating since Phase 5 (`control_logic.py:104-105`). The comment
   has been corrected — and it now points at the real remaining gap, which is
   that the **firmware** still has only a cooling branch and reports a boolean
   `hvacOn`, so heating is not expressible on the wire.

### The honest bottom line

This project demonstrates a complete, reproducible pipeline. It does **not**
demonstrate an energy saving, because nothing was metered. `docs/LIMITATIONS.md`
lists 14 limitations with measured magnitudes, and the audit fails if any
document overstates the result.

