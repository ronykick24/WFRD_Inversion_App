import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import clean_wfrd_data
from engine_wfrd import WFRD_Engine_Core

st.set_page_config(layout="wide", page_title="WFRD Geo-Master Pro")

# Paleta de Alta Visibilidad (Resistivity Standard)
GEO_PALETTE = [[0, '#000000'], [0.1, '#0000FF'], [0.3, '#00FFFF'], [0.6, '#FFFF00'], [1, '#FF0000']]

st.sidebar.title("🚀 Geonavegación Activa")
user_dip = st.sidebar.slider("Ajuste de DIP (Buzamiento)", -15.0, 15.0, 0.0, help="(-) Capas suben, (+) Capas bajan")
user_thick = st.sidebar.slider("Espesor Reservorio (ft)", 5.0, 50.0, 15.0)
n_layers = st.sidebar.selectbox("Capas", [3, 4, 5], index=2)

uploaded_file = st.file_uploader("Cargar Registro TSV", type=["tsv"])

if uploaded_file:
    df = clean_wfrd_data(pd.read_csv(uploaded_file, sep='\t'))
    engine = WFRD_Engine_Core()
    
    # 1. Inversión
    last_inc = df['INC'].iloc[-1]
    p, misfit = engine.solve(df['AD2_GW6'].values, df['MD'].values, last_inc, user_dip, n_layers)
    
    # 2. Cálculo de DTBB (Distance to Boundary)
    # Suponiendo que la capa 3 (p[2]) es nuestro objetivo
    thicknesses = p[5:9]
    interfaces = np.cumsum(np.concatenate(([0], thicknesses))) - np.sum(thicknesses)/2
    dtbb_up = abs(interfaces[1]) # Distancia al Techo
    dtbb_down = abs(interfaces[2]) # Distancia a la Base

    # --- DASHBOARD DE LABELS ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("DTBB ARRIBA (Techo)", f"{dtbb_up:.1f} ft", delta_color="inverse")
    c2.metric("DTBB ABAJO (Base)", f"{dtbb_down:.1f} ft")
    c3.metric("DIP ACTUAL", f"{user_dip}°")
    c4.metric("ESPESOR TOTAL", f"{sum(thicknesses):.1f} ft")

    # --- TRACKS VERTICALES ---
    fig_v = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=("Resistividad", "Geodirección Q"))
    for c in ['AD2_GW6', 'AU1_GW6']: fig_v.add_trace(go.Scatter(x=df[c], y=df['MD'], name=c), row=1, col=1)
    for q in ['QPD2', 'QPU1']: fig_v.add_trace(go.Scatter(x=df[q], y=df['MD'], name=q), row=1, col=2)
    fig_v.update_yaxes(autorange="reversed")
    fig_v.update_xaxes(type="log", col=1)
    fig_v.update_layout(height=400, template="plotly_dark")
    st.plotly_chart(fig_v, use_container_width=True)

    # --- CURTAIN SECTION CON LABELS DINÁMICOS ---
    tvd_grid = np.linspace(-60, 60, 150)
    # Lógica: Dip negativo -> capas suben (offset positivo en Y)
    dip_offset = -(df['MD'].values - df['MD'].values[0]) * np.tan(np.radians(user_dip))
    
    z_map = np.zeros((len(tvd_grid), len(df)))
    for j in range(len(df)):
        current_ints = interfaces + dip_offset[j]
        idx = np.searchsorted(current_ints, tvd_grid)
        z_map[:, j] = p[np.clip(idx, 0, 4)]

    fig_c = go.Figure()
    fig_c.add_trace(go.Heatmap(
        z=np.log10(z_map), x=df['MD'], y=tvd_grid,
        colorscale=GEO_PALETTE, zsmooth='best'
    ))
    
    # Línea del Pozo
    fig_c.add_trace(go.Scatter(x=df['MD'], y=np.zeros(len(df)), name="Pozo", line=dict(color='white', width=4)))

    # LABELS EN LA CORTINA
    # Marcador de Broca
    fig_c.add_annotation(x=df['MD'].iloc[-1], y=0, text="📍 BROCA", showarrow=True, arrowhead=2, arrowcolor="white", font=dict(color="white"))
    # Distancias
    fig_c.add_annotation(x=df['MD'].iloc[0], y=interfaces[1], text=f"TECHO: {dtbb_up:.1f}ft", showarrow=False, font=dict(color="cyan"))
    fig_c.add_annotation(x=df['MD'].iloc[0], y=interfaces[2], text=f"BASE: {dtbb_down:.1f}ft", showarrow=False, font=dict(color="yellow"))

    fig_c.update_layout(height=600, title=f"Sección Estructural Proactiva (NBI: {last_inc - user_dip:.1f}°)", 
                        yaxis_title="Relativo al Pozo (ft)", template="plotly_dark")
    st.plotly_chart(fig_c, use_container_width=True)
    
    if abs(dtbb_up) < 5: st.error("⚠️ ALERTA: PROXIMIDAD AL TECHO")
