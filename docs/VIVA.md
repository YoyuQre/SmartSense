# Viva Preparation

**Project:** Occupancy-Based Lighting/HVAC
Using Machine Learning
**Team 8:** 241829 Mohammed Amin Kaifi Patel, 241833 Mohammed Yahya Mohammed
Qayyum Qureshi, 241836 Abuzar Sayyed

**Read the rule first.** The brief states: *"All members must be able to
individually explain the full pipeline at viva — group work does not exempt
individual understanding."*

That means the question **"explain the project"** can be asked of **any** of
you, and "my teammate did that part" is not an answer. So:

- **Sections 1–7 below are the full-pipeline explanation. Every member must be
  able to give all of them.** They are short on purpose.
- **Section 8** is the suggested split of emphasis, if you want to lead with
  your own strongest area.
- **Sections 9–12** are the hard questions, with answers you should all know.

---

## 1. The 30-second pitch

> We built an IoT system that predicts whether a room is occupied from cheap
> sensors, and uses that to decide when lighting and HVAC would run. We trained
> an occupancy classifier and a temperature regressor on simulated data, added
> an anomaly detector because cheap sensors fail in ways a model does not
> expect, then wrapped it in a fail-safe control layer and a Streamlit
> dashboard. The main technical finding is that a Z-score detector is
> structurally blind to a stuck sensor, and a simple range-based rule fixes it.
> We did not measure any energy saving, because nothing was metered.

## 2. The pipeline, in order

Say it as a chain. If you can say this without notes, you understand the project.

```
Sensors (PIR / LDR / DS18B20, every 5 min)
   -> serial 115200 baud, or frozen CSV
   -> validation (schema, ranges, hour feature)
   -> ML inference: occupancy DecisionTree, temperature LinearRegression
   -> anomaly detection: rolling Z-score + range-based flatline
   -> sensor health: normal | suspect
   -> control: light ON/OFF, HVAC OFF/COOLING/HEATING, with fail-safe
   -> dashboard: 4 tabs, recomputes all metrics by inference
```

**One sentence per stage, ready to go:**

1. **Sensors** — PIR on GPIO 27 for motion, LDR on GPIO 34 for light, DS18B20
   on GPIO 4 for temperature, sampled every 5 minutes.
2. **Transport** — UART at 115200 baud for live, or two frozen CSVs for
   reproducibility. No cloud.
3. **Validation** — check the schema and that values are in plausible ranges;
   derive `hour` from the timestamp.
4. **Occupancy model** — a depth-3 decision tree over hour, motion, light_level,
   temperature. Outputs a probability; threshold 0.5.
5. **Temperature model** — ordinary least squares over hour, motion, light_level,
   occupancy.
6. **Anomaly detection** — two detectors. A rolling Z-score for spikes and
   drops, and a range-over-a-window rule for stuck sensors. They are separate
   because they mean different things.
7. **Sensor health** — only a *flatline* can mark a sensor `suspect`. A Z-score
   hit is a real event, so the sensor stays trusted.
8. **Control** — pure functions of (predicted occupancy, temperature, light,
   health). Light ON only if occupied, dark, and trusted. HVAC off if
   unoccupied, cooling above 26 °C, heating below 20 °C.
9. **Fail-safe** — a suspect required sensor forces its actuator off, and the
   reason is recorded.
10. **Dashboard** — four tabs, Historical Simulation by default, serial mode
    only with a real ESP32.

## 3. Why a decision tree for occupancy?

Four heterogeneous inputs, a binary target, and a deployment target that values
auditability. A depth-3 tree has **8 leaves** — you can print every branch and
show a human exactly why a decision was made. It also beat the alternatives we
tried on 5-fold cross-validation. Deeper trees scored similarly on paper and
would have been impossible to justify to an examiner or debug on a
microcontroller.

Depth and `min_samples_leaf` were chosen by grid search over
depths {2, 3, 4, 5, 6, 8, 12, unlimited} × min-leaf {1, 5, 10, 20}, 5 folds.

## 4. Why linear regression for temperature?

Temperature is a continuous quantity, and the relationship is close to additive:
occupancy adds heat, hour drives a diurnal cycle. A linear model with four
terms is the right complexity — it is auditable, it has no training
hyperparameters to get wrong, and it already explains 67 % of the variance.
The coefficients are physically sensible, which is a good sanity check:
occupancy **+1.80 °C**, hour **+0.22**, motion **+0.15**, light **+0.006**,
intercept 19.28.

## 5. The main technical finding — be ready to defend this

> **A Z-score detector cannot find a stuck sensor, and this is structural
> rather than a tuning problem.**

A Z-score is `(x − mean) / std`. When a sensor freezes, both the deviation from
the mean *and* the local standard deviation collapse toward zero. Their ratio
never grows, so the statistic never crosses any threshold. No choice of window
or `k` repairs it — you can make `k` arbitrarily small and you will simply flag
ordinary quiet periods.

Measured on the 33 scored anomalies:

| Detector | TP | FP | FN | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| rolling Z-score (inherited) | 18 | 3 | 15 | 0.857 | 0.545 | 0.667 |
| Z-score + flatline ranges (ours) | 25 | 10 | 8 | 0.714 | 0.758 | 0.735 |

Stuck sensors: **0 of 12 → 12 of 12.**

**The subtler point, which is worth the marks:** the obvious fix — "flag N
identical readings" — *also* fails, in the opposite direction. Two reasons:

- The injected faults are not exact copies. They dither by a bit:
  `[26.75, 26.72, 26.73, 26.73, 26.73]`, range 0.03 °C. An exact-equality rule
  detects **none** of them.
- Natural identical-value runs in clean data reach 3 samples for temperature,
  9 for light, 49 for motion. A short exact-equality rule fires constantly.

So detection is on the rolling **range** (max − min) over a trailing window,
with a duration and an extra condition. The light rule's **mean ≥ 300** gate is
the single most important part of it: without it, a constant reading of `0` —
which is real darkness, not a fault — makes the rule fire 18 times on ordinary
nights instead of once.

## 6. The safety argument

> A suspect sensor never drives its actuator. If the reading is untrusted, the
> actuator is forced off and the reason records which sensor was distrusted.

The reasoning, not just the rule: an untrusted reading is **insufficient
evidence to act**, not evidence for the opposite action. So a suspect
temperature forces HVAC *off* — it does not get replaced with a plausible 22 °C
and it does not get flipped to heating.

This is asserted over all 2016 replay rows, not only in unit tests, and a
companion test fails if the invariant ever passes vacuously — otherwise a bug
that made every sensor suspect would make the test pass for the wrong reason.

## 7. What we would not claim

Say these **before** you are asked. An examiner who catches you overclaiming
costs more than one who never claimed it.

- **No energy saving was measured.** Nothing was metered. 709 `COOLING`
  intervals is a count of decisions, not kilowatt-hours.
- **The data is simulated**, including the anomalies, whose ground truth we
  injected ourselves. That is the biggest methodological weakness.
- **The firmware was never run on a board.** It compiles and passes Wokwi
  lint; the Wokwi simulation needed a `WOKWI_CLI_TOKEN` we did not have.
- **The firmware does not run the models** and has no heating branch.
- **Occupancy accuracy 0.7778 beats "always empty" (0.7361) by only +0.0417.**
  That margin is modest and we say so. The temperature model is the stronger
  half.

---

## 8. Suggested emphasis per member

**Fill in your own stronger areas** — this is about what you each lead with, not
a claim about who did what. Since all three of you must be able to explain
everything above, treat this as a starting point for the demo, not a boundary.

| Member | Lead with | Be extra ready on |
| --- | --- | --- |
| 241829 Mohammed Amin Kaifi Patel | *Sensor/anomaly story* | §5 the Z-score blind spot; flatline thresholds; `light_level` is normalised 0–1000, not lux |
| 241833 Mohammed Yahya Mohammed Qayyum Qureshi | *ML story* | §3 tree choice; §4 linear model; chronological split and the 0.90 autocorrelation; why not random split |
| 241836 Abuzar Sayyed | *Control, safety and delivery* | §6 fail-safe; HVAC three-valued field; dashboard and serial gating; `final_audit.py` |

## 9. Hard questions — model and data

**Q. Is a random train/test split acceptable here?**
No. Temperature lag-1 autocorrelation is **0.90**, so a random split puts a test
row's immediate neighbour in the training set. The model would be scored on
near-copies of its training data. We cut at a single timestamp —
1440 train rows (Jan 5–9), 576 test rows (Jan 10–11) — so
`max(train) < min(test)` holds by construction.

**Q. Your test period is the weekend. Isn't that unfair?**
It is a real distribution shift, and we report it rather than hide it.
Occupancy rate is 0.5236 in training and **0.2639** in test — the model is
being asked about a quieter period than it trained on. That is exactly why we
also ran a weekday diagnostic: the same persisted model on Wednesday alone
scores F1 0.8618. We present the chronological number as the headline and the
weekday one as a diagnostic, not a better result.

**Q. Why is precision only 0.55 with recall 0.95?**
The operating point is the default 0.5 threshold and we did not tune it,
because for lighting the asymmetry is obvious: a false positive switches a light
on when nobody is there, which wastes a little; a false negative leaves someone
in the dark. We chose the cheap error. If the application were cost-sensitive
in the other direction, the threshold is the knob, not the model.

**Q. Did you check for data leakage?**
Yes, explicitly. The target is excluded from its own features in both models.
The split has no boundary overlap, and `cross_boundary_twin_mask` plus the
leakage checks in `src/preprocessing.py` are asserted by tests.

**Q. How do you know the model generalises?**
Strictly, we do not know — we can only say it generalises one week forward on
one simulated room. That is why the dashboard recomputes metrics by inference
rather than storing numbers, and why every artifact is hash-pinned. The honest
statement is that the *pipeline* is reproducible; the *performance* is
provisional.

## 10. Hard questions — anomaly detection

**Q. Your flatline thresholds look suspiciously well matched to the injected
fault. 0.03 °C exactly?**
Yes, and that is the honest weakness of this part of the project. The injected
fault's range is exactly 0.03 °C and the natural overnight minimum is also
0.03 °C, so the rule sits on the boundary. Below 0.03 the fault is invisible;
at 0.03 a few genuinely still night windows are also reported. It costs 7
false-positive rows out of 2016 (0.35 %) to detect the temperature fault at
all. A 5-sample window also works but doubles the false positives (12 → 7 at 6
samples), so 6 is the measured operating point. `sensitivity_table()` prints the
whole sweep so you can check it rather than trust it. **None of these numbers
would transfer to real hardware unchanged.**

**Q. Why is the anomaly recall only 0.758?**
All 8 remaining misses in the scored region are single-sample events — 3
`light_drop`, 4 `motion_false_trigger`, 1 `temperature_spike`. A one-sample
excursion is not a distribution shift, so a Z-score has nothing to compare it
against. **No flatline is missed.** So the detector's weakness is precisely
delimited: it is good at sustained faults and blind to single-sample spikes,
which is a different and arguably lower-severity failure.

**Q. Why exclude the first 288 rows from scoring?**
Because the rolling Z-score cannot judge a window it has not filled. 288 rows
is one day. We disclose this, and we give the alternative: scored over all 2016
rows it is TP 30 / FP 11 / FN 10, precision 0.732, recall 0.750, F1 0.741. The
choice costs about 0.008 of recall and we picked the framing that is worse for
our headline recall.

**Q. What is the difference between an anomaly and an unhealthy sensor?**
Different things, calling for different responses, so we do not merge them. A
**flatline** is a signature no real environment reproduces → sensor `suspect` →
the fail-safe may act. A **Z-score hit** is a real hot, cold or bright period →
sensor stays `normal`, still reported as an anomaly. A temperature spike is not
a broken thermometer, so treating it as one would wrongly switch the HVAC off.

**Q. Why prioritise specific reasons over generic ones?**
For diagnosis. If a row is both a flatline and a Z-score hit, reporting
`temperature_flatline` tells an engineer which rule to look at. The underlying
flags are also kept in the result so the overlap stays auditable instead of
being lost by a priority chain.

## 11. Hard questions — control, safety, systems

**Q. Why is `hvac_status` one field and not two booleans?**
So that "cooling and heating both on" is not merely forbidden by a rule but
**unrepresentable in the schema**. The thresholds are also asserted mutually
exclusive at import time, so a bad edit fails at startup rather than silently.

**Q. Heating never fires. Is it broken?**
No, and that is a property of the data. 324 rows are below 20 °C but none has
predicted occupancy 1 — the coldest predicted-occupied row is 22.47 °C. The
generator makes cold coincide with absence. The heating branch *is* implemented
and unit-tested; the replay just never reaches it. The gating counts for every
branch, including the empty ones, are printed precisely so "never triggered" is
distinguishable from "broken".

**Q. Can you show the fail-safe actually working?**
Genuinely, for HVAC: 12 rows have a suspect temperature sensor and on **7** of
those HVAC would otherwise have been cooling, so those 7 are real overrides.
The light fail-safe fires on 8 rows but is a no-op every time, because the light
fault sits at 573, above the 300 threshold, so the light would have been off
anyway. We would rather tell you that than let you discover it.

**Q. What if the occupancy model is wrong?**
Then the wrong decision is made for at most one 5-minute interval, the reason
column records it, and the anomaly layer can independently mark a sensor
suspect. Note the *direction* of the asymmetry: a false positive wastes a little
energy, a false negative leaves a person in the dark. That is why recall 0.95 is
an acceptable operating point here.

**Q. Why does serial mode show nothing without hardware?**
Because fabricating placeholder readings would make the dashboard lie. Port
candidates are filtered to ESP32 vendor IDs, Bluetooth COM ports are rejected,
and if nothing passes, no status cards render. The app tells you what it needs.

**Q. Why SHA-256 hashes?**
Because retraining would silently invalidate every number in the report. The
dashboard re-hashes each artifact at load and *reports* the result on the System
Status tab rather than raising — so a reviewer sees the drift instead of a stack
trace. And `final_audit.py` fails if a training API appears in any shipped
path. A green test suite plus matching hashes is a stronger claim than either
alone.

**Q. How would you deploy this on real hardware?**
In priority order: record real data for several weeks and re-measure every
threshold; meter the actuators so an energy claim becomes a measurement; run the
models on-device or on a gateway to close the ESP32 → model → actuator path;
implement the heating branch in firmware; calibrate `light_level` against a lux
meter; collect real anomaly labels so detection is tested on faults it was not
tuned on.

## 12. Questions you should ask them

Asking good questions is part of a strong demo.

- "Your occupancy gain over the trivial baseline is 4 points — where would you
  expect it to come from, and what would move it?"
- "If you had two more weeks, would you collect real data or improve the
  model?"
- "What is the failure mode you are most worried about in a real building?"
- "Why a 5-minute sampling interval?"
- "If the room genuinely does hold a constant temperature overnight, how would
  you distinguish that from a stuck sensor?" — *this one has no good answer in
  the current design. It needs a second, independent temperature estimate. Say
  so.*

---

## Quick self-test

Each of you should be able to answer all of these without notes:

1. The 30-second pitch.
2. The pipeline chain, stage by stage.
3. Why a chronological split and what the 0.90 autocorrelation has to do with it.
4. Why a depth-3 tree, and what the grid was.
5. Why linear regression for temperature, and the occupancy coefficient.
6. Why a Z-score cannot find a stuck sensor — the ratio argument.
7. Why range instead of repeated values, and what the light mean-gate does.
8. The difference between an anomaly and an unhealthy sensor.
9. The fail-safe, and why an untrusted reading is "insufficient evidence" rather
   than the opposite action.
10. Why `hvac_status` is one field.
11. The five things we do not claim.
12. Any single number in the report, and where it came from.

If you can answer 1–12 cold, you can do the viva.
