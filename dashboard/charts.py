"""
Charting functions for AquaSense dashboard
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def create_time_series_plot(timestamps, predictions, actuals=None):
    """Create time series plot with predictions"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=predictions,
        mode='lines',
        name='Predictions',
        line=dict(color='blue', width=2)
    ))
    
    if actuals is not None:
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=actuals,
            mode='lines',
            name='Actual',
            line=dict(color='green', width=2)
        ))
    
    fig.update_layout(
        title="Water Quality Predictions",
        xaxis_title="Time",
        yaxis_title="Quality Metric Value",
        hovermode='x unified'
    )
    
    return fig


def create_anomaly_plot(timestamps, values, anomalies):
    """Create plot highlighting detected anomalies"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=values,
        mode='markers+lines',
        name='Measurements',
        line=dict(color='blue'),
        marker=dict(size=5)
    ))
    
    anomaly_mask = anomalies == -1
    fig.add_trace(go.Scatter(
        x=timestamps[anomaly_mask],
        y=values[anomaly_mask],
        mode='markers',
        name='Anomalies',
        marker=dict(color='red', size=10, symbol='x')
    ))
    
    fig.update_layout(
        title="Anomaly Detection Results",
        xaxis_title="Time",
        yaxis_title="Measurement Value",
        hovermode='x unified'
    )
    
    return fig


def create_distribution_plot(data, title):
    """Create distribution histogram"""
    fig = px.histogram(data, nbins=30, title=title)
    return fig
