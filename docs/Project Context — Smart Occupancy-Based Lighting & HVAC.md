# **Project Context — Smart Occupancy-Based Lighting & HVAC**

## **1\. Project Overview**

We are developing a **college-level IoT \+ Machine Learning subject project** titled:

> **IoT-Based Occupancy Prediction and Intelligent Lighting/HVAC Control Using Machine Learning**

The project is intentionally designed to be **simple, understandable, implementable, and demonstrable within a few days**.

This is NOT intended to be a production-grade IoT platform, enterprise system, or research project.

The primary goal is to demonstrate the practical integration of:

* Simulated IoT sensor data  
* Data preprocessing  
* Machine Learning  
* Occupancy classification  
* Temperature prediction  
* Anomaly detection  
* Simple automation/control logic  
* Data visualization  
* A basic interactive dashboard

The implementation should prioritize **clarity and correctness over complexity**.

---

# **2\. Problem Statement**

Traditional room lighting and HVAC systems may continue operating even when a room is unoccupied, resulting in unnecessary energy consumption.

The proposed system uses simulated IoT sensor readings to determine room occupancy and environmental conditions. Machine Learning models are then used to predict occupancy and temperature, while a simple control layer decides whether lighting and HVAC should be turned ON or OFF.

The system should also identify abnormal sensor readings using anomaly detection.

---

# **3\. Core Requirements**

The project MUST satisfy the following academic requirements:

### **Requirement 1 — Simulated Sensors**

Use at least 2–3 simulated sensors relevant to the problem.

We will use exactly three:

1. **PIR Motion Sensor**  
2. **LDR/Light Sensor**  
3. **Temperature Sensor**

No physical hardware is required.

The sensor readings will be generated synthetically using Python.

---

# **4\. Sensor Definitions**

## **4.1 PIR Motion Sensor**

Represents whether movement is detected in the room.

Values:

0 → No motion  
1 → Motion detected

The simulated data should contain realistic occupancy patterns rather than completely random values.

For example, occupied periods should generally have a higher probability of motion.

---

## **4.2 LDR / Light Sensor**

Represents ambient room light.

Use a realistic continuous range such as:

0–1000 lux

The dataset should contain reasonable relationships between occupancy, time of day, and light level.

---

## **4.3 Temperature Sensor**

Represents room temperature.

Use a realistic range approximately around:

18°C–35°C

The values should have gradual changes rather than being completely random.

---

# **5\. Dataset**

Generate a synthetic dataset using Python.

Target size:

Approximately 2,000–5,000 records

Each record should contain at least:

timestamp  
motion  
light\_level  
temperature  
occupancy

Additional derived fields may be added if genuinely useful, but do NOT unnecessarily increase the dataset complexity.

The dataset should simulate realistic room behavior.

For example:

timestamp | motion | light\_level | temperature | occupancy  
\-----------------------------------------------------------  
08:00     |   0    |     85      |    23.1     |    0  
08:05     |   1    |    140      |    23.6     |    1  
08:10     |   1    |    160      |    24.0     |    1  
08:15     |   1    |    180      |    24.4     |    1  
08:20     |   0    |     90      |    24.1     |    0

The dataset must be reproducible using a fixed random seed.

---

# **6\. Machine Learning Requirements**

The project must contain at least one prediction model and one classification/anomaly model.

We will use simple models because this is a college-level project.

## **6.1 Occupancy Classification**

Use:

> **Decision Tree Classifier**

Inputs can include:

motion  
light\_level  
temperature

Target:

occupancy

Output:

0 → Unoccupied  
1 → Occupied

Evaluate using:

* Accuracy  
* Precision  
* Recall  
* F1-score  
* Confusion Matrix

---

# **7\. Temperature Prediction**

Use:

> **Linear Regression**

The model should predict room temperature based on relevant simulated sensor/environmental information.

Possible inputs:

motion  
light\_level  
occupancy

Target:

temperature

Evaluate using:

* MAE  
* RMSE  
* R²

Do not introduce LSTM or deep learning unless specifically requested later.

---

# **8\. Anomaly Detection**

Use:

> **Z-score based anomaly detection**

The objective is to identify abnormal sensor readings.

Initially apply anomaly detection to:

* Temperature  
* Light level

Example:

24.1  
24.3  
24.2  
24.4  
38.9  ← anomaly  
24.5

The system should flag unusually high or low values.

The implementation should be easy to understand and explain during a college viva.

---

# **9\. Intelligent Control Logic**

After obtaining the occupancy prediction and sensor values, implement a simple rule-based control layer.

## **Lighting**

Conceptually:

IF room is occupied  
AND light level is below threshold  
    → LIGHT ON

ELSE  
    → LIGHT OFF

## **HVAC**

Conceptually:

IF room is occupied  
AND temperature \> defined threshold  
    → HVAC ON

ELSE  
    → HVAC OFF

Thresholds should be clearly defined as configuration values rather than hidden magic numbers.

The control logic is intentionally simple.

The purpose is to demonstrate how ML predictions can be connected to an IoT-style automation system.

---

# **10\. Overall System Architecture**

The final system should follow this conceptual pipeline:

            SIMULATED IoT SENSORS  
                      │  
        ┌─────────────┼─────────────┐  
        ↓             ↓             ↓  
     Motion         Light       Temperature  
        │             │             │  
        └─────────────┼─────────────┘  
                      ↓  
               DATA GENERATION  
                      ↓  
              DATA PREPROCESSING  
                      │  
             ┌────────┴────────┐  
             ↓                 ↓  
      DECISION TREE       LINEAR REGRESSION  
      Occupancy           Temperature  
      Classification      Prediction  
             │                 │  
             └────────┬────────┘  
                      ↓  
              ANOMALY DETECTION  
                  Z-SCORE  
                      ↓  
               CONTROL ENGINE  
                ┌─────┴─────┐  
                ↓           ↓  
             LIGHTING      HVAC  
              ON/OFF       ON/OFF  
                │           │  
                └─────┬─────┘  
                      ↓  
             VISUALIZATION /  
             STREAMLIT DASHBOARD

---

# **11\. Visualization Requirements**

The project must include basic visualizations.

At minimum, provide:

1. Sensor readings over time  
2. Occupancy over time  
3. Actual vs predicted temperature  
4. Anomalous sensor readings  
5. Occupancy classification results/confusion matrix

Matplotlib is sufficient.

The graphs should have:

* Proper titles  
* Axis labels  
* Legends where appropriate  
* Readable formatting

Do not create unnecessarily complicated visualizations.

---

# **12\. Streamlit Dashboard**

A simple Streamlit dashboard is desirable.

It should display information such as:

SMART ROOM DASHBOARD

Current Occupancy: OCCUPIED

Temperature: 26.8 °C  
Light Level: 145 lux  
Motion: Detected

Lighting: ON  
HVAC: ON

Predicted Temperature: 26.5 °C

Anomaly Status: NORMAL

The dashboard should also contain relevant charts.

The dashboard is a demonstration layer, not a production application.

---

# **13\. Technology Stack**

Use only the following unless there is a clear reason to introduce something else:

### **Programming**

Python

### **Data Processing**

NumPy  
Pandas

### **Machine Learning**

Scikit-learn

### **Visualization**

Matplotlib

### **Dashboard**

Streamlit

Optional:

Seaborn  
Joblib

Do NOT introduce unnecessary frameworks such as:

* FastAPI  
* Flask  
* Django  
* Docker  
* PostgreSQL  
* Redis  
* Kafka  
* Kubernetes  
* Cloud infrastructure  
* Microservices

They are outside the scope of this project.

---

# **14\. Suggested Project Structure**

Use a clean but simple structure:

smart\_occupancy/  
│  
├── data/  
│   └── sensor\_data.csv  
│  
├── models/  
│   ├── occupancy\_model.pkl  
│   └── temperature\_model.pkl  
│  
├── src/  
│   ├── data\_generator.py  
│   ├── preprocessing.py  
│   ├── models.py  
│   ├── anomaly\_detection.py  
│   └── control\_logic.py  
│  
├── visualizations/  
│  
├── app.py  
├── requirements.txt  
└── README.md

Do not create unnecessary files simply to make the project look larger.

---

# **15\. Development Phases**

Work through the project in the following order.

## **Phase 1 — Project Definition**

Finalize:

* Project title  
* Problem statement  
* Objectives  
* Sensors  
* Dataset fields  
* ML models  
* Anomaly detection method  
* Control logic  
* Technology stack

---

## **Phase 2 — Dataset Generation**

Create the simulated sensor dataset.

Requirements:

* Realistic patterns  
* Reproducibility  
* Appropriate ranges  
* Occupied/unoccupied patterns  
* Some intentionally injected anomalies for testing

Before proceeding, inspect the generated dataset.

---

## **Phase 3 — Data Preprocessing & EDA**

Perform:

* Missing-value checking  
* Basic statistical analysis  
* Feature inspection  
* Correlation analysis  
* Basic sensor plots

---

## **Phase 4 — Machine Learning**

Implement:

### **Decision Tree**

For occupancy classification.

### **Linear Regression**

For temperature prediction.

Evaluate both models properly.

---

## **Phase 5 — Anomaly Detection**

Implement Z-score detection.

Test the system using intentionally injected abnormal readings.

---

## **Phase 6 — Control System**

Implement:

* Lighting ON/OFF  
* HVAC ON/OFF

based on occupancy and environmental conditions.

---

## **Phase 7 — Visualization**

Create the required Matplotlib visualizations.

---

## **Phase 8 — Streamlit Dashboard**

Combine the major outputs into a simple dashboard.

---

## **Phase 9 — Testing**

Verify:

* Dataset generation  
* Model training  
* Predictions  
* Anomaly detection  
* Control logic  
* Dashboard  
* Edge cases

---

## **Phase 10 — Documentation**

Prepare:

* README  
* Project report  
* System architecture  
* Methodology  
* Results  
* Screenshots  
* Conclusion  
* Future scope

---

# **16\. Important Development Rules**

These rules are VERY IMPORTANT.

### **Rule 1 — Do not overengineer**

This is a **college subject project**.

Prefer:

Simple \+ Correct \+ Explainable

over:

Complex \+ Impressive-looking \+ Difficult to explain

---

### **Rule 2 — No unnecessary technologies**

Do not add databases, APIs, cloud systems, authentication, hardware integrations, deep learning, or distributed systems unless explicitly requested.

---

### **Rule 3 — No fake results**

Do not fabricate model accuracy, RMSE, MAE, R², energy savings, or anomaly detection performance.

All reported results must come from actually running the implementation.

---

### **Rule 4 — Reproducibility**

Use fixed random seeds where appropriate so the dataset and ML results can be reproduced.

---

### **Rule 5 — Explainability**

Every major component should be understandable by a college student during a viva.

If a simpler implementation can satisfy the requirement, choose the simpler implementation.

---

### **Rule 6 — Modular implementation**

Keep data generation, preprocessing, models, anomaly detection, and control logic separated enough that they can be tested independently.

---

### **Rule 7 — Work phase-by-phase**

Do not implement the entire project in one step.

At each phase:

1. Implement only that phase.  
2. Run/test it.  
3. Inspect the result.  
4. Fix problems.  
5. Then proceed to the next phase.

Do not silently skip phases.

---

# **17\. Final Expected Outcome**

At completion, we should have a working project where:

Simulated Sensors  
       ↓  
Sensor Dataset  
       ↓  
Preprocessing  
       ↓  
Occupancy Prediction  
       \+  
Temperature Prediction  
       ↓  
Anomaly Detection  
       ↓  
Smart Lighting/HVAC Decisions  
       ↓  
Graphs  
       \+  
Streamlit Dashboard

The final project should be **small, functional, visually presentable, reproducible, and easy to defend during a college presentation/viva**.

## **Current Instruction**

Do NOT start implementing the entire project yet.

First acknowledge that you understand:

* The project objective  
* The exact sensors  
* The required ML models  
* The anomaly detection approach  
* The control logic  
* The technology constraints  
* The phase-by-phase development approach

Then wait for the instruction to begin **Phase 2 — Simulated IoT Dataset Generation**.

