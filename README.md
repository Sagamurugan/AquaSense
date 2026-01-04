# AquaSense
AquaSense – AI Powered Water Quality Monitoring & Prediction System

AquaSense is an intelligent water quality monitoring platform that predicts nutrient pollution levels in rivers using Machine Learning and Deep Learning.
It supports environmental agencies, researchers, and smart city systems by enabling early detection of pollution and assisting in decision-making for water resource management.

Project Objectives
✔ Primary Objectives

Predict Total Nitrogen (TN) concentrations using AI models

Predict Total Phosphorus (TP) in Phase-2

Analyze relationship between TN, TP, NH3, NO23, OP, and SSC

Provide a user-friendly real-time prediction dashboard

Classify water quality into:

GOOD

MODERATE

POOR

Planned AI Components
✅ Completed (Phase-1)

✔ Dataset Collection
✔ Data Cleaning & Preprocessing
✔ EDA & Trend Analysis
✔ ML Model – Random Forest
✔ Model Accuracy Achieved

TN RMSE ≈ 0.15

TN R² ≈ 0.91 (Excellent Accuracy)
✔ Streamlit Dashboard Created

Phase-2 (Upcoming Features)
🔷 1️⃣ Deep Learning – LSTM Model

To predict time-series TN levels more accurately using historical data.

Key Goals

Sequence modeling

Temporal relationship learning

Better future prediction accuracy

🔷 2️⃣ TP Prediction System

Similar model for:

Predicting Total Phosphorus

TP analysis dashboard

Use in eutrophication risk analysis

🔷 3️⃣ Anomaly Detection

Detect abnormal pollution spikes using:

Isolation Forest / AutoEncoder

Alerts for sudden contamination increase

Use Case

Disaster early warning

Industrial spill monitoring

Sewage leak detection

🔷 4️⃣ Advanced Visualization Dashboard

Multi parameter charts

Seasonal trend analysis

Correlation visualization

Comparison graphs

TN & TP over time trends

Pollution classification display

🔷 5️⃣ Real-Time Integration (Future Scope)

Connect with:

IoT river sensors

Live monitoring networks

API-based environmental data feeds


📂 Dataset Description

Dataset contains river nutrient monitoring data with:

Parameter	Meaning
dateTime	Measurement Date
TN	Total Nitrogen
TP	Total Phosphorus
NH3	Ammonia
NO23	Nitrate
OP	Orthophosphate
SSC	Suspended Sediment Concentration

🧪 Machine Learning Model
Algorithm Used

✔ Random Forest Regressor

Performance

TN RMSE: ~0.15

TN R² Score: ~0.91

This confirms the model is reliable.

🖥️ Streamlit Dashboard
Features

✔ TN Prediction Input
✔ Predict Button
✔ Water Quality Classification
✔ Color-coded Alerts
✔ Instant Output

Categories
TN (mg/L)	Status
< 1	GOOD
1 – 3	MODERATE
> 3	POOR

🧰 Tech Stack
💻 Programming

Python

📦 Libraries

Pandas

NumPy

Scikit-Learn

Matplotlib

Seaborn

Streamlit

🧠 AI Models

Random Forest (Completed)

LSTM (Upcoming)

AquaSense/
 ├── dashboard/
 │   ├── app.py
 │   └── assets/
 ├── data/
 │   ├── raw/
 │   └── processed/
 ├── models/
 │   ├── tn_random_forest_model.pkl
 │   └── tn_scaler.pkl
 ├── notebooks/
 │   ├── 01_data_cleaning.ipynb
 │   ├── 02_eda.ipynb
 │   └── 03_model_training.ipynb
 ├── scripts/
 │   └── predict.py
 ├── src/
 ├── results/
 ├── README.md
 ├── requirements.txt
 └── main.py


🚀 Real-World Applications

✔ River pollution monitoring
✔ Smart city water management
✔ Wastewater discharge monitoring
✔ Environmental policy support
✔ Academic research usage
