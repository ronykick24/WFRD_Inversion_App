import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import clean_wfrd_data
from engine_wfrd import WFRD_Engine_Core

st.set_page_config(layout="wide", page_title="WFRD Stratigraphic Navigator")

# Paleta: Azul Intenso (Arcilla) -> Cyan -> Verde -> Amarillo -> Rojo (Reservorio)
HOT_PALETTE = [[0, '#00008B'], [0.2, '#00FFFF'], [0.4, '#00FF00'], [0.7, '#FFFF00'], [1, '#FF0000']]

st.sidebar.title("🛠️ Configuración Geológica")
algo_mode = st.sidebar.selectbox("Algoritmo", ["Estocástico (Global)", "Determinístico (Local)"])
max_iters = st.sidebar.number_input("Iteraciones", 10, 500, 100)
user_dip = st.sidebar.slider("DIP [(-) Asc / (+) Desc]", -15.0, 15.0, 0.0)

uploaded_file = st.file_uploader("Subir Archivo TSV", type=["tsv"])

if uploaded_file:
    raw_df = pd.read_csv(uploaded_file, sep='\t')
    raw_df.columns = [c.upper() for c in raw_df.columns]
    df = clean_wfrd_data(raw_df)
    
    engine = WFRD_Engine_Core()
    last_inc = df['INC'].iloc[-1]
    
    # Inversión de 5 Capas definidas
    p, misfit = engine.solve(algo_mode, max_iters, df['AD2_GW6'].values, df['MD'].values, last_inc, user_dip)
    
    # --- CÁLCULO DTBB PARA CAPA 4 (RESERVORIO PRINCIPAL) ---
    thicknesses = p[5:9]
    interfaces = np.cumsum(np.concatenate(([0], thicknesses))) - np.sum(thicknesses)/2
    # El reservorio principal está entre interfaces[2] y interfaces[3]
    dtbb_top_res = abs(interfaces[2]) 
    dtbb_base_res = abs(interfaces[3])

    # --- MÉTRICAS Y ALERTAS ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("DTBB TECHO RESERVORIO", f"{dtbb_top_res:.1f} ft")
    m2.metric("DTBB BASE RESERVORIO", f"{dtbb_base_res:.1f} ft")
    m3.metric("DIP / NBI", f"{user_dip}° / {last_inc - user_dip:.1f}°")
    
    if dtbb_top_res < 5.0:
        m4.error("🚨 PROXIMIDAD AL TECHO")
    elif dtbb_base_res < 5.0:
        m4.error("🚨 PROXIMIDAD A LA BASE")
    else:
        m4.success("✅ DENTRO DE VENTANA")

    # --- TRACKS VERTICALES (Q CORREGIDAS) ---
    fig_v = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=("Resistividades (A/P)", "Geodirección (Q)"))
    for c in ['AD2_GW6', 'PD2_GW6', 'AU1_GW6']:
        if c in df.columns: fig_v.add_trace(go.Scatter(x=df[c], y=df['MD'], name=c), row=1, col=1)
    for q in ['QPD4', 'QPU1', 'QPD2']:
        if q in df.columns: fig_v.add_trace(go.Scatter(x=df[q], y=df['MD'], name=q), row=1, col=2)
    fig_v.update_yaxes(autorange="reversed")
    fig_v.update_xaxes(type="log", col=1)
    fig_v.update_layout(height=400, template="plotly_dark")
    st.plotly_chart(fig_v, use_container_width=True)

    # --- CURTAIN SECTION (DINÁMICA) ---
    tvd_grid = np.linspace(-60, 60, 150)
    dip_offset = -(df['MD'].values - df['MD'].values[0]) * np.tan(np.radians(user_dip))
    
    z_map = np.zeros((len(tvd_grid), len(df)))
    for j in range(len(df)):
        curr_ints = interfaces + dip_offset[j]
        idx = np.searchsorted(curr_ints, tvd_grid)
        z_map[:, j] = p[np.clip(idx, 0, 4)]

    fig_c = go.Figure()
    fig_c.add_trace(go.Heatmap(z=np.log10(z_map), x=df['MD'], y=tvd_grid, colorscale=HOT_PALETTE, zsmooth='best'))
    fig_c.add_trace(go.Scatter(x=df['MD'], y=np.zeros(len(df)), name="Pozo", line=dict(color='white', width=4)))
    
    # LABELS DE CAPAS
    fig_c.add_annotation(x=df['MD'].iloc[0], y=interfaces[2]+5, text="RESERVORIO PRINCIPAL", showarrow=False, font=dict(color="black", size=12))
    fig_c.add_annotation(x=df['MD'].iloc[-1], y=0, text="📍 BROCA", showarrow=True, arrowhead=2, font=dict(color="white"))

    fig_c.update_layout(height=600, title="Sección Geológica Invertida", yaxis_title="TVD Rel. (ft)", template="plotly_dark")
    st.plotly_chart(fig_c, use_container_width=True)
