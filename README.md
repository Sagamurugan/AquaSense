# AquaSense – AI Powered Water Quality Monitoring & Prediction System

## 🌊 Project Overview
AquaSense is an intelligent water quality monitoring and prediction platform that uses Machine Learning and Deep Learning to predict river water nutrient pollution.  
It assists environmental agencies, researchers, and smart city systems in **early pollution detection** and **decision-making**.

---

## 🎯 Project Objectives
- Predict **Total Nitrogen (TN)** using AI
- Predict **Total Phosphorus (TP)** in Phase-2
- Analyze relationships between TN, TP, NH3, NO23, OP, SSC
- Provide an interactive dashboard
- Classify water quality as:
  - GOOD
  - MODERATE
  - POOR

---

## ✅ Phase-1 Completed
✔ Dataset Collection  
✔ Data Cleaning & Preprocessing  
✔ EDA & Trend Analysis  
✔ Machine Learning Model Developed (Random Forest)  
✔ Streamlit Dashboard Created  

### 📌 Model Performance
| Metric | Score |
|--------|--------|
| TN RMSE | ~0.15 |
| TN R² | ~0.91 ⭐ (Excellent Accuracy) |

---

## 🔮 Phase-2 (Upcoming Features)
### 1️⃣ Deep Learning – LSTM Model
- Sequential TN Forecasting
- Long-term temporal learning

### 2️⃣ TP Prediction System
- Separate ML/DL TP Model
- TP Dashboard

### 3️⃣ Anomaly Detection
- Isolation Forest / Auto Encoder
- Detect sudden pollution spikes

### 4️⃣ Advanced Dashboard
- Multi-parameter charts
- Trends & seasonal analysis
- Alerts & classification

### 5️⃣ Future Scope
- Real-Time IoT Sensor Integration
- API Based Live River Monitoring

---

## 📂 Dataset Description
| Parameter | Meaning |
|-----------|--------|
| dateTime | Measurement Date |
| TN | Total Nitrogen |
| TP | Total Phosphorus |
| NH3 | Ammonia |
| NO23 | Nitrate |
| OP | Orthophosphate |
| SSC | Suspended Sediment Concentration |

---

## 🧠 Machine Learning Model
Algorithm:
- Random Forest Regressor

Reliable Accuracy Confirmed ✔

---

## 🖥️ Streamlit Dashboard
Features:
- TN Prediction Input
- Instant Prediction Output
- Water Quality Classification

### Classification Logic
| TN (mg/L) | Status |
|-----------|--------|
| < 1 | GOOD |
| 1–3 | MODERATE |
| > 3 | POOR |

---

## 🧰 Tech Stack
**Language**
- Python

**Libraries**
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Streamlit

**AI**
- Random Forest (Done)
- LSTM (Planned)

---

## 📁 Project Structure

AquaSense/
├── dashboard/
│ ├── app.py
│ └── assets/
├── data/
│ ├── raw/
│ └── processed/
├── models/
│ ├── tn_random_forest_model.pkl
│ └── tn_scaler.pkl
├── notebooks/
│ ├── 01_data_cleaning.ipynb
│ ├── 02_eda.ipynb
│ └── 03_model_training.ipynb
├── scripts/
│ └── predict.py
├── src/
├── results/
├── README.md
├── requirements.txt
└── main.py