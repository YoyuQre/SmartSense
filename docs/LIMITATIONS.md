# Limitations

**Project:** Occupancy-Based Lighting/HVAC
Using Machine Learning — Team 8

This file exists because a capstone report that only lists successes is not
credible. Every item below is a real property of this project, not a disclaimer
bolted on for effect. Where a limitation has a measured magnitude, the magnitude
is given.

---

## 1. No energy saving has been demonstrated

**This project does not measure, estimate, or claim an energy saving.**

The control layer decides *when* lighting and HVAC would run. It does not
measure how much energy that would use, and no actuator power, duty cycle, or
utility figure exists anywhere in the project. The 709 `COOLING` intervals are a
count of decisions, not kilowatt-hours.

The honest statement is: *occupancy-aware control is **designed to** avoid
operating lighting and HVAC in unoccupied periods, which is a necessary
precondition for reducing unnecessary energy use.* Whether it does so, by how
much, and at what comfort cost is **not** established here and would require
metered hardware.

Why it cannot be: the "actuators" are two indicator LEDs, the dataset is
simulated, and the HVAC output is a label in a CSV.

## 2. The dataset is simulated

`data/sensor_data.csv` (2016 rows, 7 days, 5-minute interval) is produced by
`src/data_generator.py`, not recorded from a physical room. Its anomalies are
**injected by the same script** with known ground-truth labels.

Consequences:

- Metrics describe the generator's realism, not a building's behaviour.
- The anomaly detector is, in part, being tested against faults whose
  characteristics were known when the thresholds were chosen. This is the
  central methodological weakness of the anomaly part of this project, and it
  is not mitigated by the flatline thresholds being principled.
- The occupancy label comes from the generator's schedule, not a ground-truth
  sensor, so label noise is absent in a way real deployments never are.

## 3. Flatline thresholds are calibrated to the simulator's noise model

Temperature uses a 6-sample window with a maximum range of 0.03 °C. The injected
fault's range is *exactly* 0.03 °C and the natural overnight minimum is *also*
0.03 °C. The rule therefore sits precisely on the boundary between the fault and
ordinary stillness, and accepts **7 false-positive rows out of 2016 (0.35%)** to
detect the temperature fault at all. The light rule adds 1 more, so flatline
rules as a whole produce **8 false-positive rows (0.40 %)**, while catching
**12 of 12** injected stuck-sensor rows.

These numbers would not transfer unchanged to real hardware. Real DS18B20 noise
at 0.1 °C resolution has a different distribution, and a real room's overnight
temperature is not genuinely static. Re-tuning on measured data is mandatory
before deployment.

## 4. Firmware has not been run on physical hardware

- The firmware **compiles** (`platformio run`: RAM 6.6 %, Flash 21.5 %) and
  **passes Wokwi lint** with 0 problems.
- The Wokwi **simulation was not executed**: it requires `WOKWI_CLI_TOKEN`,
  which was unavailable.
- No physical ESP32, PIR, LDR, or DS18B20 was connected at any point.

Consequently, unverified firmware behaviour includes: actual sensor readings,
1-Wire bus timing and its failure modes, serial framing under real baud-rate
jitter, LED states, and pin-level electrical correctness.

## 5. The ML models do not run on the microcontroller

The ESP32 firmware does not execute either model. It reads sensors, applies a
fixed threshold, and drives two LEDs. It uses **raw PIR motion** as its
occupancy signal, not the DecisionTree.

The models run on the host in Python, over either the frozen CSVs or a serial
stream. So the firmware alone cannot demonstrate the ML result, and the
end-to-end path "ESP32 → model → actuator" has not been exercised.

## 6. Heating is not expressible on the wire

The Python control layer returns a three-valued `hvac_status`
(`OFF` / `COOLING` / `HEATING`) and its heating branch is unit-tested. The
firmware has **only a cooling branch** (`temperature > 26.0`) and reports a
single boolean `hvacOn`. Heating is therefore unrepresentable in serial output
and undrivable by the current firmware.

This was found during the Phase 7 architecture review and the stale comment in
`src/config.py` that claimed heating was unimplemented has been corrected.

## 7. Two control rules are not exercised by the replay

`hvac_status` contains **709 `COOLING`, 1307 `OFF`, and 0 `HEATING`** across all
2016 rows.

This is a property of the data, not a broken rule: 324 rows fall below 20 °C,
but **none** of them has predicted occupancy 1 — the coldest predicted-occupied
row is 22.47 °C. The generator makes cold coincide with absence.

Similarly, the **light** fail-safe fires on 8 rows but is a no-op every time:
the light fault sits at light_level 573, above the 300 threshold, so a
healthy-sensor rule would have left the light off regardless. Only the
**temperature** fail-safe produces genuine overrides — 7 intervals where HVAC
was forced off and would otherwise have been cooling.

So the replay genuinely exercises the lighting rule and the HVAC fail-safe. The
**heating rule and the light fail-safe are supported by unit tests only.** The
gating counts for every branch, including the empty ones, are printed precisely
so this is distinguishable from a bug.

## 8. Occupancy generalisation is weak, and the split reveals why

Primary test metrics: accuracy 0.7778, precision 0.5451, recall 0.9539,
**F1 0.6938** (n = 576).

Three honest observations:

- **The gain over the trivial baseline is small.** Always predicting
  "unoccupied" scores accuracy 0.7361. The model improves that by **+0.0417**,
  which is not a large margin.
- **The test period is distribution-shifted.** Occupancy rate is 0.5236 in the
  training window and **0.2639** in the test window, because the chronological
  split puts the weekend last. The model is being asked about a quieter period
  than it trained on.
- **The operating point favours recall over precision** (121 false positives vs
  7 false negatives). For lighting, a false positive switches a light on when
  nobody is there — wasteful but harmless. A false negative leaves someone in
  the dark. That asymmetry is a deliberate design choice, not a tuned optimum,
  and the threshold is not tuned at all; it is the default 0.5.

The much higher weekday figure (F1 0.8618) comes from evaluating the same
persisted primary model on Wednesday alone. It is a **diagnostic, not a headline
result** — it is one day, and the model never saw it.

## 9. A split-day protocol is deliberate, and it constrains the result

Temperature lag-1 autocorrelation is **0.90**, so a random split would place a
test row's immediate neighbour in the training set and hand the model its own
answer. The chronological split is the correct choice and is the right
engineering call — but it means the test set is a *future* period, so any
concept drift is charged entirely to the model. With only 7 days simulated, that
effect cannot be separated from genuine model error.

## 10. `light_level` is not calibrated lux

`light_level` is a **normalised 0–1000 scale** produced by `map()` from the raw
ADC count. The firmware says so in its own boot banner. The 300 lighting
threshold is a threshold on that normalised scale, not on physical lux, so it
has no absolute meaning across different LDRs or different ADC references. Real
deployment needs calibration against a reference instrument.

## 11. Anomaly scoring excludes a warm-up window

Detection metrics are scored on rows **288–2015** (33 of the 40 injected
anomalies). The 288 leading rows are excluded because the inherited rolling
Z-score cannot judge a window it has not filled.

Scored over **all 2016 rows** instead, the detector is TP 30 / FP 11 / FN 10 —
precision 0.732, **recall 0.750**, F1 0.741. So excluding the warm-up costs
about 0.008 of recall, and the reported headline (0.714 / 0.758 / 0.735) is the
*more conservative* of the two on recall and the less conservative on precision.

Both framings are defensible; only the scored one is quoted, and the choice is
disclosed here. The 7 anomalies inside the warm-up split 5 detected / 2 missed.

## 12. Dashboard serial mode is untested against real hardware

Serial mode is implemented and its parser is unit-tested against synthetic
frames, and it correctly refuses to display status without an ESP32. But no
live serial session has been run, so port enumeration, driver behaviour, and
timing on a real machine are unverified. The Bluetooth COM-port rejection is
tested with faked port lists, not real hardware.

## 13. Single-room, single-building, 7 days

All results come from one simulated room over 7 days. The weekday model
diagnostic exists precisely because weekly structure matters, so 7 days is known
to be too short to characterise weekly occupancy. Nothing here demonstrates
transfer to another building, another sensor set, or another season.

## 14. Test coverage is logic coverage, not statistical validation

75 tests pass. They assert invariants — the fail-safe never acts on a suspect
sensor, the HVAC field cannot hold both states, the split has no boundary
leakage, the replay is byte-reproducible. They do **not** establish that the
models will perform this well on unseen real data. A green suite means the
specified behaviour holds, nothing more.

---

## What would raise confidence, in priority order

1. **Record real data** from a physical ESP32 in an occupied room for several
   weeks, and re-measure every threshold on it.
2. **Meter the actuators** — a smart plug on the HVAC branch — so an energy claim
   becomes a measurement instead of an intention.
3. **Run the models on-device**, or on a local gateway, so the ESP32 → model →
   actuator path is closed.
4. **Implement the heating branch in firmware** and widen `hvac_status` to three
   states on the wire.
5. **Re-tune flatline thresholds** on measured noise, and add a
   temperature-inferred cross-check so a genuinely static room can be
   distinguished from a stuck probe.
6. **Calibrate `light_level`** against a lux meter.
7. **Collect real anomaly labels** so detection is evaluated against faults it
   was not tuned on.
