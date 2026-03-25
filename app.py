import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils import clean_wfrd_data
from engine_wfrd import WFRD_Engine_Core

st.set_page_config(layout="wide", page_title="WFRD Geo-Simulator 2026")

# --- SIDEBAR: CONTROLES DE NAVEGACIÓN ---
st.sidebar.title("🎮 Configuración de Inversión")
mode = st.sidebar.selectbox("Modo de Cálculo", ["Estocástico", "Determinístico"])
n_layers = st.sidebar.slider("Capas del Modelo", 3, 5, 5)
iters = st.sidebar.number_input("Iteraciones / Max NFEV", 10, 500, 100)
ani_fix = st.sidebar.slider("Anisotropía (λ)", 1.0, 4.0, 1.5)

st.sidebar.markdown("---")
st.sidebar.title("📈 Selección de Curvas")
available_curves = ['AD2_GW6', 'AD4_GW6', 'AU1_GW6', 'PD2_GW6', 'PU1_GW6']
selected_curves = st.sidebar.multiselect("Curvas para el Track", available_curves, default=['AD2_GW6', 'AU1_GW6'])

uploaded_file = st.file_uploader("Cargar Registro TSV", type=["tsv"])

if uploaded_file:
    df = clean_wfrd_data(pd.read_csv(uploaded_file, sep='\t'))
    engine = WFRD_Engine_Core()
    
    # Datos de Trayectoria
    last_inc = df['INC'].iloc[-1]
    user_dip = st.slider("Ajustar Buzamiento (Dip)", -15.0, 15.0, 0.0)
    
    # Cálculo del NBI en ángulo
    nbi_angle = last_inc - user_dip
    
    # Ejecutar Inversión
    p, misfit = engine.solve(mode, df[selected_curves[0]].values, df['MD'].values, last_inc, user_dip, n_layers, iters)

    # --- MÉTRICAS DE GEONAVEGACIÓN ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("NBI (Ángulo Relativo)", f"{nbi_angle:.1f}°")
    c2.metric("Inclinación (Inc)", f"{last_inc:.1f}°")
    c3.metric("Misfit", f"{misfit:.4f}")
    c4.metric("λ Invertido", f"{p[9]:.2f}")

    # --- TRACK HORIZONTAL PERSONALIZADO ---
    fig_h = go.Figure()
    for curve in selected_curves:
        if curve in df.columns:
            fig_h.add_trace(go.Scatter(x=df['MD'], y=df[curve], name=curve))
    fig_h.update_layout(height=250, yaxis_type="log", title="Track de Resistividad Seleccionado", template="plotly_dark")
    st.plotly_chart(fig_h, use_container_width=True)

    # --- CURTAIN SECTION DINÁMICA ---
    tvd_grid = np.linspace(-60, 60, 150)
    thicknesses = p[5:9]
    interfaces = np.cumsum(np.concatenate(([0], thicknesses))) - np.sum(thicknesses)/2
    
    # El Dip afecta la inclinación visual de las capas
    z_map = np.zeros((len(tvd_grid), len(df)))
    dip_offset = (df['MD'].values - df['MD'].values[0]) * np.tan(np.radians(user_dip))
    
    for j in range(len(df)):
        current_ints = interfaces + dip_offset[j]
        idx = np.searchsorted(current_ints, tvd_grid)
        z_map[:, j] = p[np.clip(idx, 0, 4)]

    fig_c = go.Figure()
    fig_c.add_trace(go.Heatmap(
        z=np.log10(z_map), x=df['MD'], y=tvd_grid,
        colorscale='Viridis', zsmooth='best'
    ))

    # Trayectoria del pozo fija en 0 para ver cómo las capas "suben o bajan" respecto a él
    fig_c.add_trace(go.Scatter(x=df['MD'], y=np.zeros(len(df)), name="Wellbore", line=dict(color='white', width=3)))

    fig_c.update_layout(height=500, title=f"Sección de Cortina (Dip: {user_dip}°, NBI: {nbi_angle:.1f}°)", 
                        yaxis_title="Relativo al Pozo (ft)", template="plotly_dark")
    st.plotly_chart(fig_c, use_container_width=True)
