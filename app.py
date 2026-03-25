import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import clean_wfrd_data
from engine_wfrd import WFRD_Engine_Core

st.set_page_config(layout="wide", page_title="Earth Model Geosteer")

# --- INTERFAZ Y SELECCIÓN DE PALETA ---
st.sidebar.title("🌍 Modelo Tierra & Textura")
color_theme = st.sidebar.selectbox("Paleta de Resistividad", ["Turbo", "Electric", "Viridis", "Hot"])
n_layers = st.sidebar.slider("Capas del Modelo", 3, 9, 5)
calc_mode = st.sidebar.radio("Motor de Inversión", ["Estocástico Global", "Determinístico"])
user_dip = st.sidebar.slider("DIP (Buzamiento)", -15.0, 15.0, 0.0)

uploaded_file = st.file_uploader("Cargar TSV", type=["tsv"])

if uploaded_file:
    raw_df = pd.read_csv(uploaded_file, sep='\t')
    raw_df.columns = [c.upper() for c in raw_df.columns]
    df = clean_wfrd_data(raw_df)
    
    engine = WFRD_Engine_Core()
    last_inc = df['INC'].iloc[-1]
    
    # Inversión
    p, misfit = engine.solve(calc_mode, 100, df['AD2_GW6'].values, df['MD'].values, last_inc, user_dip, n_layers)
    
    res_vals = p[:n_layers]
    thick_vals = p[n_layers:2*n_layers-1]
    interfaces = np.cumsum(np.concatenate(([0], thick_vals))) - np.sum(thick_vals)/2
    
    # --- LÓGICA DE DTTB/DTBB DINÁMICA ---
    # Posición actual del pozo (0 ft en el eje TVD relativo)
    # Buscamos en qué capa está el 0 (nuestro pozo)
    current_layer_idx = np.searchsorted(interfaces, 0) - 1
    current_layer_idx = np.clip(current_layer_idx, 0, n_layers-1)
    
    dttb = abs(interfaces[current_layer_idx])      # Distancia al límite superior
    dtbb = abs(interfaces[current_layer_idx + 1])  # Distancia al límite inferior

    # --- MÉTRICAS ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"DTTB (Capa {current_layer_idx+1})", f"{dttb:.1f} ft", delta="Superior")
    c2.metric(f"DTBB (Capa {current_layer_idx+1})", f"{dtbb:.1f} ft", delta="Inferior")
    c3.metric("Resistividad Local", f"{res_vals[current_layer_idx]:.1f} Ωm")
    c4.metric("NBI", f"{last_inc - user_dip:.1f}°")

    # --- CURTAIN SECTION CON TEXTURA ---
    st.subheader("🌐 Sección de Cortina (Modelo de Tierra)")
    tvd_grid = np.linspace(-60, 60, 200)
    dip_offset = -(df['MD'].values - df['MD'].values[0]) * np.tan(np.radians(user_dip))
    
    z_map = np.zeros((len(tvd_grid), len(df)))
    for j in range(len(df)):
        curr_ints = interfaces + dip_offset[j]
        idx = np.searchsorted(curr_ints, tvd_grid)
        z_map[:, j] = res_vals[np.clip(idx, 0, n_layers-1)]

    # Añadir "textura" (ruido gaussiano sutil para parecer roca)
    texture = np.random.normal(0, 0.02, z_map.shape)
    z_map_textured = np.log10(z_map) + texture

    fig_c = go.Figure()
    fig_c.add_trace(go.Heatmap(
        z=z_map_textured, x=df['MD'], y=tvd_grid, 
        colorscale=color_theme, zsmooth='best', showscale=True
    ))

    # Dibujar TODAS las interfaces detectadas
    for i, inter in enumerate(interfaces):
        fig_c.add_trace(go.Scatter(
            x=df['MD'], y=np.full(len(df), inter) + dip_offset,
            mode='lines', line=dict(color='rgba(255,255,255,0.3)', width=1),
            name=f"Límite {i}"
        ))

    # Pozo y Etiquetas de distancias en la broca
    fig_c.add_trace(go.Scatter(x=df['MD'], y=np.zeros(len(df)), name="Wellbore", line=dict(color='white', width=3)))
    
    # Anotaciones dinámicas de DTTB/DTBB
    fig_c.add_annotation(x=df['MD'].iloc[-1], y=interfaces[current_layer_idx], text=f"DTTB:{dttb:.1f}", showarrow=True, arrowhead=1)
    fig_c.add_annotation(x=df['MD'].iloc[-1], y=interfaces[current_layer_idx+1], text=f"DTBB:{dtbb:.1f}", showarrow=True, arrowhead=1)

    fig_c.update_layout(height=650, template="plotly_dark", yaxis_title="TVD Relativo (ft)")
    st.plotly_chart(fig_c, use_container_width=True)

    # --- TRACKS DE GEODIRECCIÓN Q ---
    st.subheader("📉 Registros GuideWave (Q)")
    fig_q = go.Figure()
    for q in ['QPD2', 'QPD4', 'QPU1', 'QPU4']:
        if q in df.columns: fig_q.add_trace(go.Scatter(x=df['MD'], y=df[q], name=q))
    fig_q.update_layout(height=300, template="plotly_dark", xaxis_title="MD (ft)")
    st.plotly_chart(fig_q, use_container_width=True)
