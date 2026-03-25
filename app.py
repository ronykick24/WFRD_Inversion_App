import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import clean_wfrd_data
from engine_wfrd import WFRD_Engine_Core

st.set_page_config(layout="wide", page_title="WFRD Geo-Master Expert")

# Paleta de alta visibilidad para identificar el Pay Zone
GEO_PALETTE = [[0, '#000000'], [0.2, '#0000FF'], [0.4, '#00FFFF'], [0.7, '#FFFF00'], [1, '#FF0000']]

# --- SIDEBAR: CONFIGURACIÓN DEL USUARIO ---
st.sidebar.title("🛠️ Configuración del Motor")
algo_mode = st.sidebar.selectbox("Algoritmo de Inversión", ["Estocástico (Global)", "Determinístico (Local)"])
max_iters = st.sidebar.number_input("Cantidad de Iteraciones", 10, 1000, 100)
n_layers = st.sidebar.slider("Capas en el Modelo", 3, 5, 5)

st.sidebar.markdown("---")
st.sidebar.title("🎮 Parámetros de Campo")
user_dip = st.sidebar.slider("DIP (Buzamiento) [(-) Asc / (+) Desc]", -15.0, 15.0, 0.0)

uploaded_file = st.file_uploader("Cargar Registro TSV", type=["tsv"])

if uploaded_file:
    raw_df = pd.read_csv(uploaded_file, sep='\t')
    raw_df.columns = [c.upper() for c in raw_df.columns] # Normalizar nombres (QPd4 -> QPD4)
    df = clean_wfrd_data(raw_df)
    
    engine = WFRD_Engine_Core()
    last_inc = df['INC'].iloc[-1]
    
    # Ejecutar Inversión
    p, misfit = engine.solve(algo_mode, max_iters, df['AD2_GW6'].values, df['MD'].values, last_inc, user_dip, n_layers)
    
    # Cálculo de DTBB (Distancia a Fronteras)
    thicknesses = p[5:9]
    interfaces = np.cumsum(np.concatenate(([0], thicknesses))) - np.sum(thicknesses)/2
    dtbb_up, dtbb_down = abs(interfaces[1]), abs(interfaces[2])

    # --- MÉTRICAS SUPERIORES ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("DTBB ARRIBA", f"{dtbb_up:.1f} ft", delta="TECHO", delta_color="inverse")
    c2.metric("DTBB ABAJO", f"{dtbb_down:.1f} ft", delta="BASE")
    c3.metric("NBI (INC-DIP)", f"{last_inc - user_dip:.1f}°")
    c4.metric("ERROR (MISFIT)", f"{misfit:.4f}")

    # --- TRACKS VERTICALES ---
    st.subheader("📊 Registros de Herramienta")
    fig_v = make_subplots(rows=1, cols=3, shared_yaxes=True, 
                          subplot_titles=("Atenuación (A)", "Fase (P)", "Geodirección (Q)"))
    
    # Atenuación (AD2, AD4, AU1...)
    for c in ['AD2_GW6', 'AD4_GW6', 'AU1_GW6']:
        if c in df.columns: fig_v.add_trace(go.Scatter(x=df[c], y=df['MD'], name=c), row=1, col=1)
    
    # Fase (PD2, PD4, PU1...)
    for c in ['PD2_GW6', 'PD4_GW6', 'PU1_GW6']:
        if c in df.columns: fig_v.add_trace(go.Scatter(x=df[c], y=df['MD'], name=c), row=1, col=2)

    # Geodirección (QPd4, QPu1...)
    for q in ['QPD2', 'QPD4', 'QPU1', 'QPU4']:
        if q in df.columns: fig_v.add_trace(go.Scatter(x=df[q], y=df['MD'], name=q), row=1, col=3)
        
    fig_v.update_yaxes(autorange="reversed")
    fig_v.update_xaxes(type="log", col=1); fig_v.update_xaxes(type="log", col=2)
    fig_v.update_layout(height=450, template="plotly_dark")
    st.plotly_chart(fig_v, use_container_width=True)

    # --- CURTAIN SECTION (DINÁMICA) ---
    st.subheader(f"🌐 Sección de Cortina Proactiva | Dip: {user_dip}°")
    tvd_grid = np.linspace(-60, 60, 150)
    # Dip negativo (-) -> Capas suben visualmente
    dip_offset = -(df['MD'].values - df['MD'].values[0]) * np.tan(np.radians(user_dip))
    
    z_map = np.zeros((len(tvd_grid), len(df)))
    for j in range(len(df)):
        current_ints = interfaces + dip_offset[j]
        idx = np.searchsorted(current_ints, tvd_grid)
        z_map[:, j] = p[np.clip(idx, 0, 4)]

    fig_c = go.Figure()
    fig_c.add_trace(go.Heatmap(z=np.log10(z_map), x=df['MD'], y=tvd_grid, colorscale=GEO_PALETTE, zsmooth='best'))
    fig_c.add_trace(go.Scatter(x=df['MD'], y=np.zeros(len(df)), name="Pozo", line=dict(color='white', width=4)))
    
    # Labels en tiempo real
    fig_c.add_annotation(x=df['MD'].iloc[-1], y=5, text=f"TOP: {dtbb_up:.1f}ft", showarrow=False, font=dict(color="white"))
    fig_c.add_annotation(x=df['MD'].iloc[-1], y=-5, text=f"BASE: {dtbb_down:.1f}ft", showarrow=False, font=dict(color="white"))

    fig_c.update_layout(height=550, yaxis_title="TVD Relativo (ft)", template="plotly_dark")
    st.plotly_chart(fig_c, use_container_width=True)
