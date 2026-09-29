# **PHASE 1 — Project Definition & Setup**

## **1\. Final Project Title**

**IoT-Based Occupancy Prediction and Intelligent Lighting/HVAC Control Using Machine Learning**

Short name for the project:

> **Smart Occupancy-Based Lighting & HVAC System**

---

## **2\. Problem Statement**

Traditional lighting and HVAC systems may continue operating even when a room is unoccupied, resulting in unnecessary energy consumption.

This project develops a simulated IoT-based system that uses **motion, light, and temperature sensors** to determine room occupancy and intelligently control lighting and HVAC operation.

---

## **3\. Objectives**

We'll keep only the objectives that we actually implement:

1. Generate **simulated IoT sensor data** for a room.  
2. Predict whether the room is **occupied or unoccupied**.  
3. Predict room temperature using a **Linear Regression model**.  
4. Detect abnormal sensor readings using **Z-score anomaly detection**.  
5. Automatically determine:  
   * 💡 Lighting ON/OFF  
   * ❄️ HVAC ON/OFF  
6. Visualize sensor data, predictions, anomalies and control decisions.  
7. Create a simple **Streamlit dashboard**.

---

# **4\. Simulated Sensors**

We'll use exactly **3 sensors**.

### **Sensor 1 — PIR Motion Sensor**

Measures whether movement is detected.

0 → No motion  
1 → Motion detected

### **Sensor 2 — LDR Light Sensor**

Measures ambient light.

0–1000 lux

Higher value → brighter room.

### **Sensor 3 — Temperature Sensor**

Measures room temperature.

Approximately 18–35 °C  
---

# **5\. Dataset**

We'll generate our own synthetic dataset.

For example:

timestamp  
motion  
light\_level  
temperature  
occupancy

Example:

| Time | Motion | Light | Temperature | Occupancy |
| ----- | ----- | ----- | ----- | ----- |
| 08:00 | 0 | 85 | 23.1 | 0 |
| 08:05 | 1 | 140 | 23.6 | 1 |
| 08:10 | 1 | 160 | 24.0 | 1 |
| 08:15 | 1 | 180 | 24.4 | 1 |
| 08:20 | 0 | 90 | 24.1 | 0 |

We'll generate **\~2,000–5,000 records**. That's more than enough for a college project.

---

# **6\. ML Models**

We don't need complicated models.

### **Model 1 — Occupancy Classification**

**Decision Tree Classifier**

Input:

Motion  
Light Level  
Temperature

Output:

0 → Unoccupied  
1 → Occupied

We'll evaluate it using:

* Accuracy  
* Precision  
* Recall  
* F1-score  
* Confusion Matrix

---

### **Model 2 — Temperature Prediction**

**Linear Regression**

Inputs:

Motion  
Light Level  
Occupancy

Output:

Predicted Temperature

We'll evaluate:

* MAE  
* RMSE  
* R²

This satisfies the required **prediction model**.

---

# **7\. Anomaly Detection**

We'll use:

### **Z-Score**

Simple and very easy to explain.

For example, if temperature suddenly becomes:

24.1  
24.3  
24.2  
24.4  
38.9 ← anomaly  
24.5

the system flags the abnormal reading.

We'll perform anomaly detection on:

* Temperature  
* Light level

---

# **8\. Smart Control Logic**

After the ML prediction, we'll apply simple control rules.

### **Lighting**

IF room is occupied  
AND light level is low  
→ LIGHT ON

ELSE  
→ LIGHT OFF

### **HVAC**

IF room is occupied  
AND temperature \> 26°C  
→ HVAC ON

ELSE  
→ HVAC OFF

This gives us an actual **IoT automation component**, rather than just ML.

---

# **9\. Final System Flow**

             SIMULATED IoT SENSORS  
                       │  
          ┌────────────┼────────────┐  
          ↓            ↓            ↓  
       Motion        Light      Temperature  
          │            │            │  
          └────────────┼────────────┘  
                       ↓  
                DATA PREPROCESSING  
                       │  
              ┌────────┴────────┐  
              ↓                 ↓  
       DECISION TREE      LINEAR REGRESSION  
       Occupancy          Temperature  
       Prediction         Prediction  
              │                 │  
              └────────┬────────┘  
                       ↓  
                ANOMALY DETECTION  
                    Z-SCORE  
                       │  
                       ↓  
                CONTROL ENGINE  
                ┌──────┴──────┐  
                ↓             ↓  
             LIGHTING        HVAC  
              ON/OFF         ON/OFF  
                │             │  
                └──────┬──────┘  
                       ↓  
                STREAMLIT DASHBOARD  
---

# **10\. Technology Stack**

Keep it simple:

Language       → Python

Data           → Pandas, NumPy

ML             → Scikit-learn

Visualization  → Matplotlib

Dashboard      → Streamlit

Dataset        → Synthetic / simulated

No database, Docker, FastAPI, cloud deployment, IoT hardware, or complicated architecture is necessary.

---

# **11\. Project Folder**

We'll eventually create:

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

But **don't create all of this yet**.