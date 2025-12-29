"""
PROJECT: Predictive Maintenance & Visual Analytics
PURPOSE: Translating model outputs into multi-dimensional stakeholder visualizations.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# 1. Load data
df = pd.read_csv('output/final_dashboard_data.csv')

# --- 3D MAP ---
df_sorted = df.sort_values(by='Actual_Failure')
# Updated column names (no brackets)
fig_3d = px.scatter_3d(df_sorted, 
                       x='Air temperature K', 
                       y='Torque Nm', 
                       z='Rotational speed rpm',
                       color='Actual_Failure',
                       color_continuous_scale=['#2E91E5', '#E15F23'], 
                       title='3D Operational Risk Map',
                       opacity=0.7)

# FIX: Responsive Stretching
fig_3d.update_layout(autosize=True, margin=dict(l=0, r=0, b=0, t=50))

# --- RADAR CHART ---
# Updated list to match the sanitized CSV columns
categories = ['Air temperature K', 'Process temperature K', 'Rotational speed rpm', 'Torque Nm', 'Tool wear min']

def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

df_norm = df[categories].apply(normalize)
df_norm['Actual_Failure'] = df['Actual_Failure']

healthy_avg = df_norm[df_norm['Actual_Failure'] == 0][categories].mean().tolist()
failure_avg = df_norm[df_norm['Actual_Failure'] == 1][categories].mean().tolist()

healthy_avg += [healthy_avg[0]]
failure_avg += [failure_avg[0]]
categories_loop = categories + [categories[0]]

fig_radar = go.Figure()
fig_radar.add_trace(go.Scatterpolar(r=healthy_avg, theta=categories_loop, fill='toself', name='Healthy System'))
fig_radar.add_trace(go.Scatterpolar(r=failure_avg, theta=categories_loop, fill='toself', name='Failure Signature'))

# FIXED: Responsive "Stretching" Layout for Radar Chart
fig_radar.update_layout(
    title="Failure 'DNA' Profile",
    autosize=True,
    # This reduces the empty padding (l=left, r=right, b=bottom, t=top)
    margin=dict(l=80, r=80, b=80, t=100), 
    polar=dict(
        radialaxis=dict(visible=True, range=[0, 1]),
        # Ensures the chart takes up the maximum circular space
        angularaxis=dict(direction="clockwise") 
    ),
    # Moves the legend out of the way so the chart can grow
    legend=dict(
        orientation="v",
        yanchor="middle",
        y=0.5,
        xanchor="left",
        x=1.1
    )
)

fig_3d.show()
fig_radar.show()