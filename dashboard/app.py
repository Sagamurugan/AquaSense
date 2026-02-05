import streamlit as st
import numpy as np
import joblib
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Set Page Config
st.set_page_config(page_title="AquaSense Dashboard", layout="wide", page_icon="💧")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    /* Force text colors for the white card background */
    [data-testid="stMetricLabel"] {
        color: #444444 !important;
    }
    [data-testid="stMetricValue"] {
        color: #000000 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Main Header
col1, col2 = st.columns([1, 4])
with col1:
    st.image("https://img.icons8.com/color/96/000000/water-element.png", width=80) 
with col2:
    st.title("AquaSense AI Monitor")
    st.markdown("### Intelligent Water Quality Prediction & Anomaly Detection")

st.divider()

# Load Models
@st.cache_resource
def load_models():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, "..", "models")
    
    models = {}
    try:
        models['tn_model'] = joblib.load(os.path.join(models_dir, "tn_random_forest_model.pkl"))
        models['tn_scaler'] = joblib.load(os.path.join(models_dir, "tn_scaler.pkl"))
        models['tp_model'] = joblib.load(os.path.join(models_dir, "tp_random_forest_model.pkl"))
        models['tp_scaler'] = joblib.load(os.path.join(models_dir, "tp_scaler.pkl"))
        models['anomaly_model'] = joblib.load(os.path.join(models_dir, "anomaly_isolation_forest.pkl"))
    except Exception as e:
        st.error(f"Error loading models: {e}")
    return models

models = load_models()

# Sidebar Inputs
with st.sidebar:
    st.header("🎛️ Control Panel")
    st.markdown("Adjust water parameters:")
    
    NH3 = st.slider("Ammonia (NH3)", 0.0, 5.0, 0.03, 0.01)
    NO23 = st.slider("Nitrate (NO23)", 0.0, 10.0, 1.2, 0.1)
    OP = st.slider("Orthophosphate (OP)", 0.0, 5.0, 0.07, 0.01)
    SSC = st.slider("Suspended Sediment (SSC)", 0.0, 1000.0, 180.0, 10.0)
    
    run_btn = st.button("🚀 Analyze Quality", type="primary")

# Main Dashboard Logic
if run_btn and models:
    
    # --- 1. CORE PREDICTION PIPELINE ---
    
    # Predict TN
    tn_input = np.array([[NH3, NO23, OP, SSC]])
    tn_scaled = models['tn_scaler'].transform(tn_input)
    tn_pred_raw = models['tn_model'].predict(tn_scaled)
    tn_val = float(tn_pred_raw[0] if tn_pred_raw.ndim == 1 else tn_pred_raw[0][0])
    
    # Predict TP (Using TN)
    tp_input = np.array([[tn_val, NH3, NO23, OP, SSC]])
    tp_scaled = models['tp_scaler'].transform(tp_input)
    tp_pred_raw = models['tp_model'].predict(tp_scaled)
    tp_val = float(tp_pred_raw[0])
    
    # Anomaly Detection
    anomaly_input = np.array([[tn_val, tp_val, NH3, NO23, OP, SSC]])
    anomaly_pred = models['anomaly_model'].predict(anomaly_input)[0]
    
    # --- 2. DISPLAY RESULTS ---
    
    # Top Metrics Row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted Nitrogen (TN)", f"{tn_val:.3f} mg/L", delta="Main Pollutant")
    c2.metric("Predicted Phosphorus (TP)", f"{tp_val:.3f} mg/L", help="Experimental Prediction")
    
    status_text = "NORMAL" if anomaly_pred == 1 else "ANOMALY"
    status_color = "normal" if anomaly_pred == 1 else "inverse"
    c3.metric("System Status", status_text, delta_color=status_color)
    
    # Classification Logic
    if tn_val < 1:
        quality_label = "GOOD"
        gauge_color = "green"
    elif tn_val < 3:
        quality_label = "MODERATE"
        gauge_color = "orange"
    else:
        quality_label = "POOR"
        gauge_color = "red"
    
    c4.metric("Quality Class", quality_label)

    st.divider()

    # --- 3. ADVANCED VISUALIZATIONS ---
    
    col_viz1, col_viz2 = st.columns([1, 2])
    
    # A. Gauge Chart
    with col_viz1:
        st.subheader("📊 Quality Gauge")
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = tn_val,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "TN Level (mg/L)"},
            gauge = {
                'axis': {'range': [None, 5.0]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 1], 'color': "lightgreen"},
                    {'range': [1, 3], 'color': "orange"},
                    {'range': [3, 5], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': tn_val
                }
            }
        ))
        fig_gauge.update_layout(height=400, margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # B. Sensitivity Analysis
    with col_viz2:
        st.subheader("� Sensitivity Analysis: Effect of Ammonia on TN")
        
        # Simulate varying NH3
        nh3_range = np.linspace(0.0, 5.0, 50)
        sim_tn_values = []
        
        for sim_nh3 in nh3_range:
            # Create input vector with VARYING NH3 but CONSTANT others
            sim_input = np.array([[sim_nh3, NO23, OP, SSC]]) 
            sim_scaled = models['tn_scaler'].transform(sim_input)
            pred = models['tn_model'].predict(sim_scaled)
            sim_tn_values.append(float(pred[0] if pred.ndim == 1 else pred[0][0]))
            
        # Create Line Chart
        df_sens = pd.DataFrame({"Ammonia (NH3)": nh3_range, "Predicted TN": sim_tn_values})
        fig_line = px.line(df_sens, x="Ammonia (NH3)", y="Predicted TN", 
                          title="How Increasing Ammonia Increases Nitrogen Pollution",
                          markers=True)
        fig_line.add_vline(x=NH3, line_dash="dash", line_color="red", annotation_text="Current Value")
        fig_line.update_layout(height=400)
        st.plotly_chart(fig_line, use_container_width=True)

else:
    if not models:
        st.warning("⚠️ Models not found. Calculating...")
    else:
        st.info("👈 Adjust parameters in the sidebar and click **Analyze Quality** to start.")

# Footer
st.markdown("---")
st.markdown("© 2024 AquaSense AI | Powered by Random Forest & Streamlit")
