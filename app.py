import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils import clean_wfrd_data
from engine_wfrd import WFRD_Advanced_Engine

st.set_page_config(layout="wide", page_title="WFRD Geosteering AI")

# Paleta Geológica Pro: Azul (Agua), Gris (Roca), Oro (Oil), Rojo (Gas)
RESERVOIR_PALETTE = [
    [0.0, 'rgb(0, 0, 100)'], [0.2, 'rgb(100, 200, 255)'], 
    [0.5, 'rgb(255, 255, 255)'], [0.8, 'rgb(255, 215, 0)'], [1.0, 'rgb(150, 0, 0)']
]

st.sidebar.title("🎮 Parámetros de Simulación")
user_inc = st.sidebar.slider("Inclinación de la Broca (°)", 75.0, 95.0, 85.0)
user_dip = st.sidebar.slider("Ajuste de Dip (Buzamiento)", -10.0, 10.0, 0.0)
show_labels = st.sidebar.checkbox("Mostrar Etiquetas de Distancia", True)

uploaded_file = st.file_uploader("Subir Archivo TSV de Weatherford", type=["tsv"])

if uploaded_file:
    df = clean_wfrd_data(pd.read_csv(uploaded_file, sep='\t'))
    engine = WFRD_Advanced_Engine()
    
    with st.spinner('Procesando inversión estocástica...'):
        p, misfit = engine.solve(df['AD2_GW6'].values, df['MD'].values, user_inc, user_dip)
    
    # --- CÁLCULO DE DISTANCIAS ---
    thicknesses = p[5:9]
    interfaces = np.cumsum(np.concatenate(([0], thicknesses))) - np.sum(thicknesses)/2
    dist_to_top = abs(interfaces[2]) # Ejemplo: Capa 3 es el Reservorio

    # --- MÉTRICAS Y ALERTAS ---
    m1, m2, m3 = st.columns(3)
    m1.metric("Misfit (Error)", f"{misfit:.4f}")
    m2.metric("Distancia al Techo", f"{dist_to_top:.1f} ft")
    if dist_to_top < 6.0:
        m3.error("🚨 ALERTA: PROXIMIDAD AL TECHO")
    else:
        m3.success("🛡️ DENTRO DE PAY-ZONE")

    # --- TRACKS HORIZONTALES (GR + RES) ---
    fig_t = go.Figure()
    # Simulamos GR basado en litología de la inversión
    gr_val = 150 - (np.log10(df['AD2_GW6']) * 40)
    fig_t.add_trace(go.Scatter(x=df['MD'], y=gr_val, name="Gamma Ray", line=dict(color='green')))
    fig_t.add_trace(go.Scatter(x=df['MD'], y=df['AD2_GW6'], name="Resistividad", yaxis="y2", line=dict(color='red')))
    
    fig_t.update_layout(
        height=300, template="plotly_dark",
        yaxis=dict(title="GR (API)", side="left"),
        yaxis2=dict(title="Res (Ωm)", side="right", overlaying="y", type="log"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_t, use_container_width=True)

    # --- CURTAIN SECTION (SHADING REAL) ---
    tvd_grid = np.linspace(-60, 60, 150)
    z_map = np.zeros((len(tvd_grid), len(df)))
    for i, z in enumerate(tvd_grid):
        idx = np.searchsorted(interfaces, z)
        z_map[i, :] = p[min(idx, 4)]

    fig_c = go.Figure()
    fig_c.add_trace(go.Heatmap(
        z=np.log10(z_map), x=df['MD'], y=tvd_grid,
        colorscale=RESERVOIR_PALETTE, zsmooth='best'
    ))

    # Trayectoria dinámica
    well_y = (df['MD'] - df['MD'].min()) * np.tan(np.radians(user_inc - 90 - user_dip))
    fig_c.add_trace(go.Scatter(x=df['MD'], y=well_y, name="Trayectoria", line=dict(color='white', width=4)))

    if show_labels:
        fig_c.add_annotation(x=df['MD'].iloc[-1], y=well_y[-1], text="BROCA", showarrow=True, arrowhead=1)

    fig_c.update_layout(height=600, title="Simulación Estructural 50ft (1D/2D)", yaxis_title="TVD Relativo (ft)")
    st.plotly_chart(fig_c, use_container_width=True)
