import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from engine_wfrd import StochasticInversion5L

st.set_page_config(layout="wide")
COLOR_SCALE = [[0, 'darkblue'], [0.2, 'blue'], [0.5, 'white'], [0.8, 'orange'], [1, 'red']]

st.title("🛰️ Inversión Estocástica Multicapa WFRD (5 Capas)")

# Controles de perforación en tiempo real
col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
with col_ctrl1:
    inc_bit = st.slider("Inclinación de Broca (INC)", 80.0, 95.0, 85.0)
with col_ctrl2:
    dip_form = st.slider("Dip de la Formación", -10.0, 10.0, 2.0)
with col_ctrl3:
    misfit_limit = st.text("Misfit de Validación", "Calculando...")

uploaded_file = st.file_uploader("Cargar Datos TSV", type=["tsv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep='\t').iloc[1:].apply(pd.to_numeric, errors='coerce').dropna(subset=['MD'])
    engine = StochasticInversion5L()
    
    # Ejecutar inversión para el tramo visible
    params, misfit_val = engine.run_inversion(df['AD2_GW6'].values, df['MD'].values, inc_bit, dip_form)
    st.sidebar.write(f"Misfit Final: {misfit_val:.4f}")

    # --- TRACKS HORIZONTALES (SENSIBILIDAD) ---
    st.subheader("Tracks de Sensibilidad: Sensores GuideWave (Arriba/Abajo)")
    fig_tracks = go.Figure()
    fig_tracks.add_trace(go.Scatter(x=df['MD'], y=df['AD2_GW6'], name="33ft - Cercano", line=dict(color='cyan')))
    fig_tracks.add_trace(go.Scatter(x=df['MD'], y=df['AU1_GW6'], name="50ft - Profundo", line=dict(color='magenta')))
    fig_tracks.update_layout(height=300, margin=dict(t=0, b=0))
    st.plotly_chart(fig_tracks, use_container_width=True)

    # --- CURTAIN SECTION MULTICAPA ---
    # Reconstrucción de la formación 5 capas
    tvd_grid = np.linspace(-60, 60, 120)
    md_range = df['MD'].values
    z_curtain = np.zeros((len(tvd_grid), len(md_range)))
    
    # Lógica de capas basada en parámetros invertidos
    interfaces = np.cumsum(params[5:]) - np.sum(params[5:])/2
    for i, z in enumerate(tvd_grid):
        idx = np.searchsorted(interfaces, z)
        z_curtain[i, :] = params[idx]

    fig_curtain = go.Figure(data=go.Heatmap(
        z=z_curtain, x=md_range, y=tvd_grid, colorscale=COLOR_SCALE, zsmooth='best'
    ))

    # Trayectoria dinámica que cruza capas
    path_tvd = (md_range - md_range[0]) * np.tan(np.radians(inc_bit - 90))
    fig_curtain.add_trace(go.Scatter(x=md_range, y=path_tvd, name="Trayectoria Real", line=dict(color='black', width=4)))

    fig_curtain.update_layout(height=600, title="Sección Estructural Invertida (5 Capas Físicas)")
    st.plotly_chart(fig_curtain, use_container_width=True)
