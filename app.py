import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import clean_wfrd_data
from engine_wfrd import WFRD_Engine_Core

st.set_page_config(layout="wide", page_title="WFRD Geo-Expert System")

# Colores consistentes para curvas
CURVE_COLORS = {
    'AD2_GW6': '#FF0000', 'AD4_GW6': '#FF4500', 'AU1_GW6': '#FF8C00',
    'PD2_GW6': '#0000FF', 'PD4_GW6': '#1E90FF', 'PU1_GW6': '#87CEEB',
    'QPD2': '#00FF00', 'QPU1': '#32CD32'
}

st.sidebar.title("⚙️ Configuración")
calc_mode = st.sidebar.selectbox("Método", ["Estocástico", "Determinístico"])
n_layers = st.sidebar.slider("Capas", 3, 5, 5)
iters = st.sidebar.slider("Iteraciones", 50, 500, 100)

uploaded_file = st.file_uploader("Cargar TSV", type=["tsv"])

if uploaded_file:
    df = clean_wfrd_data(pd.read_csv(uploaded_file, sep='\t'))
    engine = WFRD_Engine_Core()
    
    # Inputs de Geodirección
    user_dip = st.sidebar.slider("Buzamiento (Dip)", -15.0, 15.0, 0.0)
    last_inc = df['INC'].iloc[-1]
    nbi_angle = last_inc - user_dip
    
    # Inversión
    p, misfit = engine.solve(calc_mode, df['AD2_GW6'].values, df['MD'].values, last_inc, user_dip, n_layers, iters)

    # --- TRACKS VERTICALES (LITOLOGÍA Y GEODIRECCIÓN) ---
    st.subheader("📊 Registros de Perforación y Geodirección")
    
    fig_v = make_subplots(rows=1, cols=3, shared_yaxes=True, 
                          subplot_titles=("Amplitud (dB)", "Fase (deg)", "Geodirección (Q)"))
    
    # Track 1: Amplitud
    for c in ['AD2_GW6', 'AD4_GW6', 'AU1_GW6']:
        if c in df.columns:
            fig_v.add_trace(go.Scatter(x=df[c], y=df['MD'], name=c, line=dict(color=CURVE_COLORS[c])), row=1, col=1)
    
    # Track 2: Fase
    for c in ['PD2_GW6', 'PD4_GW6', 'PU1_GW6']:
        if c in df.columns:
            fig_v.add_trace(go.Scatter(x=df[c], y=df['MD'], name=c, line=dict(color=CURVE_COLORS[c])), row=1, col=2)

    # Track 3: Geodirección
    for c in ['QPD2', 'QPD4', 'QPU1']:
        if c in df.columns:
            fig_v.add_trace(go.Scatter(x=df[c], y=df['MD'], name=c, line=dict(width=2)), row=1, col=3)

    fig_v.update_yaxes(autorange="reversed", title="Profundidad (MD)")
    fig_v.update_xaxes(type="log", col=1)
    fig_v.update_layout(height=500, template="plotly_dark", showlegend=True)
    st.plotly_chart(fig_v, use_container_width=True)

    # --- CURTAIN SECTION CON LOGICA DE DIP ---
    st.subheader(f"🌐 Sección de Cortina (NBI: {nbi_angle:.1f}°)")
    
    tvd_grid = np.linspace(-60, 60, 150)
    thicknesses = p[5:9]
    interfaces = np.cumsum(np.concatenate(([0], thicknesses))) - np.sum(thicknesses)/2
    
    z_map = np.zeros((len(tvd_grid), len(df)))
    dip_offset = (df['MD'].values - df['MD'].values[0]) * np.tan(np.radians(user_dip))
    
    for j in range(len(df)):
        current_ints = interfaces + dip_offset[j]
        idx = np.searchsorted(current_ints, tvd_grid)
        z_map[:, j] = p[np.clip(idx, 0, 4)]

    fig_c = go.Figure()
    fig_c.add_trace(go.Heatmap(
        z=np.log10(z_map), x=df['MD'], y=tvd_grid,
        colorscale='Turbo', zsmooth='best', colorbar=dict(title="Res")
    ))
    
    # Trayectoria fija para ver movimiento relativo
    fig_c.add_trace(go.Scatter(x=df['MD'], y=np.zeros(len(df)), name="Pozo", line=dict(color='white', width=3)))
    
    fig_c.update_layout(height=500, yaxis_title="TVD Relativo (ft)", xaxis_title="MD (ft)", template="plotly_dark")
    st.plotly_chart(fig_c, use_container_width=True)
    
    # Alertas de proximidad
    st.info(f"NBI: {nbi_angle:.1f}° | Misfit: {misfit:.4f} | Anisotropía λ: {p[9]:.2f}")
