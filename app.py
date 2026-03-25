import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils import clean_wfrd_data
from engine_wfrd import WFRD_Engine_Core

st.set_page_config(layout="wide", page_title="WFRD Geo-Pilot v2")

# --- SIDEBAR: CONFIGURACIÓN AVANZADA ---
st.sidebar.title("🕹️ Control de Inversión")
calc_mode = st.sidebar.radio("Método de Ajuste", ["Estocástico", "Determinístico"])
n_layers = st.sidebar.selectbox("Modelo de Capas", [3, 4, 5], index=2)
max_iters = st.sidebar.slider("Iteraciones Máximas", 10, 300, 100)

st.sidebar.markdown("---")
st.sidebar.title("📊 Selección de Canales")
canales = ['AD2_GW6', 'AD4_GW6', 'AU1_GW6', 'PD2_GW6', 'PU1_GW6']
selec = st.sidebar.multiselect("Curvas a Visualizar", canales, default=['AD2_GW6', 'AU1_GW6'])

uploaded_file = st.file_uploader("Cargar Datos TSV", type=["tsv"])

if uploaded_file:
    df = clean_wfrd_data(pd.read_csv(uploaded_file, sep='\t'))
    engine = WFRD_Engine_Core()
    
    # 1. Parámetros de Trayectoria Real
    last_inc = df['INC'].iloc[-1]
    last_azm = df['AZM'].iloc[-1] if 'AZM' in df.columns else 0.0
    
    # 2. Control interactivo de DIP para calcular NBI
    user_dip = st.slider("Ajustar Buzamiento de Capa (Dip)", -15.0, 15.0, 0.0)
    nbi_angle = last_inc - user_dip # El NBI es el ángulo relativo de corte
    
    # 3. Ejecutar Inversión basada en la primera curva seleccionada
    p, misfit = engine.solve(calc_mode, df[selec[0]].values, df['MD'].values, last_inc, user_dip, n_layers, max_iters)

    # --- DASHBOARD DE MÉTRICAS ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("NBI (Relativo)", f"{nbi_angle:.1f}°")
    c2.metric("Inclinación Real", f"{last_inc:.1f}°")
    c3.metric("Azimut", f"{last_azm:.1f}°")
    c4.metric("Misfit", f"{misfit:.4f}")

    # --- TRACKS HORIZONTALES INDIVIDUALES ---
    fig_h = go.Figure()
    for c in selec:
        fig_h.add_trace(go.Scatter(x=df['MD'], y=df[c], name=c))
    fig_h.update_layout(height=250, yaxis_type="log", title="Resistividades WFRD", template="plotly_dark")
    st.plotly_chart(fig_h, use_container_width=True)

    # --- SECCIÓN DE CORTINA DINÁMICA ---
    tvd_grid = np.linspace(-60, 60, 150)
    thicknesses = p[5:9]
    interfaces = np.cumsum(np.concatenate(([0], thicknesses))) - np.sum(thicknesses)/2
    
    # Simulación de capas inclinadas por el DIP
    z_map = np.zeros((len(tvd_grid), len(df)))
    dip_offset = (df['MD'].values - df['MD'].values[0]) * np.tan(np.radians(user_dip))
    
    for j in range(len(df)):
        current_ints = interfaces + dip_offset[j]
        idx = np.searchsorted(current_ints, tvd_grid)
        z_map[:, j] = p[np.clip(idx, 0, 4)]

    fig_c = go.Figure()
    fig_c.add_trace(go.Heatmap(
        z=np.log10(z_map), x=df['MD'], y=tvd_grid,
        colorscale='Geyser', zsmooth='best'
    ))
    
    # Trayectoria del pozo (fija en 0 para ver el cruce relativo)
    fig_c.add_trace(go.Scatter(x=df['MD'], y=np.zeros(len(df)), name="Wellbore", line=dict(color='black', width=4)))

    fig_c.update_layout(height=550, title=f"Sección Estructural (NBI: {nbi_angle:.1f}°)", 
                        yaxis_title="TVD Relativo (ft)", template="plotly_white")
    st.plotly_chart(fig_c, use_container_width=True)
