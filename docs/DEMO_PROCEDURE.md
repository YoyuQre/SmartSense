# Demo Procedure

**Project:** Occupancy-Based Lighting/HVAC
Using Machine Learning — Team 8
**Duration:** 10–12 minutes
**Hardware required:** none. The demo runs entirely from frozen files.

The demo is designed so that it cannot fail on stage. It needs no ESP32, no
network, and no live data. Every number shown is reproducible from
`data/sensor_data.csv` and `data/control_output.csv`, both SHA-256 pinned.

---

## Before the demo (5 minutes, do this first)

Open a terminal **in the project folder** and run:

```bash
pip install -r requirements.txt
python -m pytest tests -q
```

Expected: **`75 passed`**. If this does not say 75 passed, stop and fix it
before presenting. Do not demo a red test suite.

Optionally warm the dashboard so the first paint is instant:

```bash
streamlit run app.py
```

Then close it and proceed to the demo proper.

---

## Act 1 — Reproducibility first (90 seconds)

Open a terminal in the project folder and run:

```bash
python src/final_audit.py
```

Expected: **`RESULT: 44/44 checks passed`**.

This single command is the strongest thing you can show, because it proves
every claim in the report is currently true. It verifies:

- the dataset and both models are byte-identical to the frozen Phase 2/4
  artifacts (SHA-256),
- the 75-test suite is green,
- the firmware still compiles,
- the fail-safe invariant holds across all 2016 replay rows,
- the report, dashboard, and this document all exist,
- and no document claims an energy saving that was never measured.

**Say this:** *"Everything I'm about to show you is checked by that one command.
Nothing is hand-copied into a slide."*

## Act 2 — The dashboard (4 minutes)

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. The health endpoint returns `200 ok`.

Confirm the mode selector reads **Historical Simulation** (it is the default and
is labelled prominently).

### 2a. System Status

Point out:

- 2016 rows, 7 days, 5-minute interval.
- The model identities and their **SHA-256 hashes shown as `unchanged`**.

**Say this:** *"The dashboard reports the hash of every file it reads, so you
can see it is reading the same bytes the models were trained on."*

### 2b. ML Performance

This is the honest part — show it without editing.

- Occupancy: accuracy **0.7778**, precision **0.5451**, recall **0.9539**,
  F1 **0.6938**.
- Temperature: MAE **1.3404 °C**, RMSE **1.9532 °C**, R² **0.6715**.
- The always-unoccupied baseline is **0.7361** accuracy, so the model's margin
  is **+0.0417**.

**Say this, unprompted:** *"Accuracy only beats 'always empty' by four points.
Precision is 0.55 — the model over-predicts occupancy, which is the right
trade-off for lighting: switching a light on when nobody is there wastes a
little, leaving someone in the dark is worse. The temperature model is the
stronger half of the project — it beats a constant-mean predictor by a large
margin, 1.95 °C RMSE against 3.44 °C."*

If asked why the weekday F1 is higher (0.8618): that is the *same persisted
model* evaluated on Wednesday alone, one day, a diagnostic for the weekend
distribution shift — not a separate better model.

### 2c. Sensor & Anomaly Monitoring

- Anomaly markers on the time series.
- The sensor-health event list: temperature suspect 12 rows, light suspect 8,
  motion 0.
- Click **Anomaly only** in the sidebar.

**Say this:** *"There are two different signals here, and they are deliberately
not merged. A flatline is a signature no real environment produces, so that
sensor is marked suspect. A Z-score hit is a genuine hot or bright period, so
the sensor is still trusted — a temperature spike is not a broken thermometer."*

### 2d. Intelligent Control

- The decision table: lighting OFF 1732 / ON 284; HVAC OFF 1307 / COOLING 709 /
  **HEATING 0**.
- The `control_reason` column on each row.

**Say this about HEATING 0, before anyone asks:** *"Heating never fires in this
replay, and that is a property of the data rather than a bug. 324 rows are
below 20 °C, but none of them has predicted occupancy 1 — the generator makes
cold coincide with absence. The heating rule is covered by unit tests, and the
replay prints the gating count for every branch including the empty ones, so
you can tell the difference between 'never triggered' and 'broken'."*

## Act 3 — The finding worth the most marks (90 seconds)

The strongest technical result in this project is a negative one. Show it:

> A Z-score detector **cannot** find a stuck sensor. A frozen value drives both
> the local standard deviation *and* the deviation from the baseline toward
> zero, so the ratio a Z-score tests never grows. No choice of window or
> threshold `k` fixes it — this is structural, not a tuning problem.

Measured, on the 33 scored anomalies:

| Detector | TP | FP | FN | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| rolling Z-score (inherited) | 18 | 3 | 15 | 0.857 | 0.545 | 0.667 |
| Z-score + flatline ranges (ours) | 25 | 10 | 8 | 0.714 | 0.758 | 0.735 |

- Stuck sensors detected: **0 of 12 → 12 of 12.**
- Cost: 7 extra false positives. All 8 remaining misses are single-sample
  events (3 `light_drop`, 4 `motion_false_trigger`, 1 `temperature_spike`).
- The flatline rule tests the **range** (max − min) over a window, not repeated
  values, because the injected faults dither:
  `[26.75, 26.72, 26.73, 26.73, 26.73]` is a stuck sensor, not a constant.

## Act 4 — The safety invariant (45 seconds)

> **A suspect sensor never drives its actuator.** If a reading is untrusted, its
> actuator is forced OFF and the reason records which sensor was distrusted. The
> bad value is never replaced with a plausible-looking number — an untrusted
> reading is treated as *insufficient evidence to act*, not as evidence for the
> opposite action.

Verified across all 2016 replay rows, and a companion test fails if the check
ever passes vacuously.

20 replay rows name a suspect sensor in `control_reason`:

- **12** rows have a suspect temperature sensor, which forces HVAC off. On **7**
  of those, HVAC would otherwise have been cooling — those 7 are genuine
  overrides.
- **8** rows have a suspect light sensor, which forces lighting off. On **0**
  of those is it a genuine override, because the light fault sits at
  light_level 573, above the 300 threshold, so a healthy-sensor rule would have
  left the light off anyway.

Worth stating rather than hiding: the fail-safe is genuinely exercised 7 times
for HVAC, and the 8 light cases are a no-op that happens to be safe. The
invariant holds either way, and a test asserts it either way — that is the
point of asserting an invariant rather than a count.

## Act 5 — The firmware (60 seconds)

```bash
python -m platformio run
```

Expected: **SUCCESS**, RAM 6.6 %, Flash 21.5 %.

**State this clearly, do not overclaim:**

> The firmware compiles and passes Wokwi lint with zero problems. We did not run
> it on a physical ESP32 — we had no board available — and we did not run the
> Wokwi simulation because it needs a `WOKWI_CLI_TOKEN` we did not have. So the
> firmware is *verified to build and lint*, not verified to run.

If asked what the firmware does: it reads PIR, LDR and DS18B20 every 5 minutes,
applies a fixed threshold, drives two indicator LEDs, and prints
`DATA,millis,motion,light_level,temperature,light_status,hvac_status` at 115200
baud. It **does not run the ML models** — those run on the host. And it has
only a cooling branch; heating exists in the Python layer but is not yet
expressible in serial output.

## Act 6 — Close honestly (30 seconds)

> To summarise what this project does and does not show: it shows a reproducible
> pipeline — simulated data, trained models, anomaly detection that fixes a real
> structural blind spot in Z-scores, deterministic control with a fail-safe, and
> a dashboard that recomputes every metric by inference. It does **not** show a
> measured energy saving, because no actuator was ever metered. That is the
> first thing we would fix, and it is written up in `docs/LIMITATIONS.md`.

---

## Anticipated questions with short answers

**"Did you save energy?"**
No, and we do not claim it. The control layer decides *when* lighting and HVAC
would run; it does not measure what that costs. The 709 `COOLING` intervals are
a count of decisions, not kilowatt-hours. Occupancy-aware control is a necessary
condition for avoiding unnecessary operation, not proof of it.

**"Is the data real?"**
No — simulated by `src/data_generator.py`, including the injected anomalies
with known ground truth. That is the biggest methodological weakness, and it
means the anomaly thresholds are partly tuned to faults whose characteristics we
already knew.

**"Why 5-minute sampling?"**
It is fixed in `src/config.py` as the single source of truth, and every
threshold is quoted in both samples and minutes so it cannot be misread.

**"Why not a random train/test split?"**
Temperature lag-1 autocorrelation is 0.90, so a random split puts a test row's
immediate neighbour in the training set and hands the model its own answer. We
cut at a single timestamp, so `max(train) < min(test)` holds by construction.

**"Why such a shallow tree?"**
Depth 3, min-leaf 20, 8 leaves — selected by 5-fold cross-validation over
depths {2,3,4,5,6,8,12,∞}. A shallow tree is deliberate for an embedded target:
it is auditable, and you can print every branch.

**"Is 0.7778 accuracy good?"**
It beats the trivial 0.7361 baseline by 4 points, which is modest, and we say so.
The stronger result here is the temperature model (RMSE 1.95 °C versus 3.44 °C
for a constant-mean predictor). The occupancy problem on 7 simulated days is
genuinely hard, and our own weekend split exposes that.

---

## If something breaks

| Symptom | Do this |
| --- | --- |
| `pytest` not 75 passed | Stop. Fix before presenting. |
| `final_audit.py` below 44/44 | Read the FAILURES list at the bottom; it names the file. |
| Dashboard will not start | `streamlit run app.py --server.port 8502` |
| `streamlit` not found | `pip install -r requirements.txt` |
| Port 8501 busy | Use `--server.port 8502` and say so. |
| Asked to show live hardware | Say plainly it was never run on a board, then show the replay. Do not improvise. |
