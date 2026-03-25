import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import clean_wfrd_data
from engine_wfrd import WFRD_Engine_Core

st.set_page_config(layout="wide", page_title="WFRD Geosteering Expert")

st.sidebar.title("🎮 Controles Proactivos")
user_dip = st.sidebar.slider("Buzamiento (Dip) [(-) Asc / (+) Desc]", -15.0, 15.0, 0.0)
calc_mode = st.sidebar.selectbox("Algoritmo", ["Estocástico", "Determinístico"])
n_layers = st.sidebar.number_input("Capas", 3, 5, 5)

uploaded_file = st.file_uploader("Cargar Datos (.tsv)", type=["tsv"])

if uploaded_file:
    df = clean_wfrd_data(pd.read_csv(uploaded_file, sep='\t'))
    engine = WFRD_Engine_Core()
    last_inc = df['INC'].iloc[-1]
    
    # Inversión Estocástica
    p, misfit = engine.solve(calc_mode, df['AD2_GW6'].values, df['MD'].values, last_inc, user_dip, n_layers, 100)

    # --- TRACKS DE GEODIRECCIÓN (Q) Y RESISTIVIDAD ---
    st.subheader("📊 Registros Verticales y Respuesta Q")
    fig_v = make_subplots(rows=1, cols=3, shared_yaxes=True, 
                          subplot_titles=("Resistividad (Ohm-m)", "Geodirección (Q)", "Inclinación"))

    # Resistividades
    for c in ['AD2_GW6', 'AU1_GW6', 'PD2_GW6']:
        if c in df.columns:
            fig_v.add_trace(go.Scatter(x=df[c], y=df['MD'], name=c), row=1, col=1)

    # Curvas Q (Lógica: Negativo = Abajo, Positivo = Arriba)
    for q in ['QPD2', 'QPD4', 'QPU1']:
        if q in df.columns:
            fig_v.add_trace(go.Scatter(x=df[q], y=df['MD'], name=q, line=dict(dash='dot')), row=1, col=2)
            # Línea de referencia cero para Q
            fig_v.add_vline(x=0, line_dash="dash", line_color="gray", row=1, col=2)

    fig_v.add_trace(go.Scatter(x=df['INC'], y=df['MD'], name="INC", line=dict(color='white')), row=1, col=3)
    
    fig_v.update_yaxes(autorange="reversed")
    fig_v.update_xaxes(type="log", col=1)
    fig_v.update_layout(height=450, template="plotly_dark")
    st.plotly_chart(fig_v, use_container_width=True)

    # --- CURTAIN SECTION (ESTRUCTURA INCLINADA) ---
    st.subheader(f"🌐 Sección Estructural | Dip: {user_dip}° (NBI: {last_inc - user_dip:.1f}°)")
    
    tvd_grid = np.linspace(-60, 60, 150)
    thicknesses = p[5:9]
    interfaces = np.cumsum(np.concatenate(([0], thicknesses))) - np.sum(thicknesses)/2
    
    # Lógica de dibujo: Dip Negativo -> Offset sube con el MD
    # Usamos -tan(dip) para que el signo negativo resulte en ascenso visual
    dip_offset = -(df['MD'].values - df['MD'].values[0]) * np.tan(np.radians(user_dip))
    
    z_map = np.zeros((len(tvd_grid), len(df)))
    for j in range(len(df)):
        current_ints = interfaces + dip_offset[j]
        idx = np.searchsorted(current_ints, tvd_grid)
        z_map[:, j] = p[np.clip(idx, 0, 4)]

    fig_c = go.Figure()
    fig_c.add_trace(go.Heatmap(
        z=np.log10(z_map), x=df['MD'], y=tvd_grid,
        colorscale='Turbo', zsmooth='best'
    ))
    
    # Trayectoria del pozo fija en 0
    fig_c.add_trace(go.Scatter(x=df['MD'], y=np.zeros(len(df)), name="Wellbore", line=dict(color='white', width=4)))
    
    # Alerta visual de Q
    last_q = df['QPD2'].iloc[-1] if 'QPD2' in df.columns else 0
    q_dir = "CAPA POR DEBAJO" if last_q < 0 else "CAPA POR ARRIBA"
    
    fig_c.update_layout(height=550, yaxis_title="TVD Relativo (ft)", template="plotly_dark")
    st.plotly_chart(fig_c, use_container_width=True)
    
    st.info(f"Detección Q: {q_dir} (Valor: {last_q:.2f}) | Misfit: {misfit:.4f}")
