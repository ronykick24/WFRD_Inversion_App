import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils import clean_wfrd_data
from engine_wfrd import StochasticInversion5L

st.set_page_config(layout="wide", page_title="WFRD Inversion 5L")

# Colores: Azul (Conductivo), Blanco (Neutral), Rojo (Resistivo)
COLOR_SCALE = [[0, 'navy'], [0.3, 'skyblue'], [0.5, 'white'], [0.7, 'orange'], [1, 'darkred']]

st.title("🛰️ WFRD High-Angle Inversion: 5 Capas & Anisotropía")

uploaded_file = st.file_uploader("Cargar Datos (.tsv)", type=["tsv"])

if uploaded_file:
    df = clean_wfrd_data(pd.read_csv(uploaded_file, sep='\t'))
    
    # Controles Dinámicos
    c1, c2, c3 = st.columns(3)
    inc_val = c1.slider("Inclinación Real (Inc)", 70.0, 95.0, 85.0)
    dip_val = c2.slider("Buzamiento (Dip)", -10.0, 10.0, 0.0)
    
    # Procesar Inversión
    engine = StochasticInversion5L()
    # Tomamos un segmento para optimizar velocidad
    params, misfit_val = engine.run_inversion(df['AD2_GW6'].values, df['MD'].values, inc_val, dip_val)
    
    c3.metric("Misfit (Error)", f"{misfit_val:.4f}", delta_color="inverse")

    # --- TRACKS DE SENSIBILIDAD ---
    st.subheader("Sensibilidad de Sensores: 33ft vs 50ft")
    fig_t = go.Figure()
    fig_t.add_trace(go.Scatter(x=df['MD'], y=df['AD2_GW6'], name="Cercano (33ft)", line=dict(color='cyan')))
    fig_t.add_trace(go.Scatter(x=df['MD'], y=df['AU1_GW6'], name="Profundo (50ft)", line=dict(color='magenta')))
    fig_t.update_layout(height=250, template="plotly_dark", margin=dict(t=10, b=10))
    st.plotly_chart(fig_t, use_container_width=True)

    # --- CURTAIN SECTION (SECCIÓN ESTRUCTURAL) ---
    st.subheader("Sección Estructural: Cruce de Capas")
    
    # Crear malla de 5 capas basada en espesores invertidos
    tvd_grid = np.linspace(-60, 60, 100)
    interfaces = np.cumsum(params[5:9]) - np.sum(params[5:9])/2
    
    z_map = np.zeros((len(tvd_grid), len(df)))
    for i, z in enumerate(tvd_grid):
        layer_idx = np.searchsorted(interfaces, z)
        z_map[i, :] = params[layer_idx]

    fig_c = go.Figure(data=go.Heatmap(
        z=z_map, x=df['MD'], y=tvd_grid, colorscale=COLOR_SCALE, zsmooth='best'
    ))

    # Trayectoria calculada con la Inclinación (INC)
    path_tvd = (df['MD'] - df['MD'].iloc[0]) * np.tan(np.radians(inc_val - 90))
    fig_c.add_trace(go.Scatter(x=df['MD'], y=path_tvd, name="Pozo", line=dict(color='black', width=4)))

    fig_c.update_layout(height=500, yaxis_title="TVD Relativo (ft)", xaxis_title="MD (ft)")
    st.plotly_chart(fig_c, use_container_width=True)
