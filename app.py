import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from engine_wfrd import WFRD_Engine_Core

st.set_page_config(layout="wide", page_title="Geo-Mapper Pro V5")

# --- SELECTOR DE TEMPLATE Y COLORES ---
st.sidebar.title("📋 Template & Visual")
with st.sidebar.expander("Configurar Canales", expanded=True):
    res_ch = st.selectbox("Inversión Principal", ["AD2_GW6", "PD2_GW6", "AD4_GW6"])
    q_chs = st.multiselect("Canales Q", ["QPD2", "QPD4", "QPU1", "QPU4"], default=["QPD2", "QPU1"])
    color_theme = st.selectbox("Paleta de Tierra", ["Turbo", "Electric", "Hot", "Cividis"])

st.sidebar.title("🔭 Proyección Proactiva")
user_dip = st.sidebar.slider("DIP Formación (°)", -15.0, 15.0, 0.0)
future_inc = st.sidebar.slider("Proyectar Inclinación (°)", 80.0, 100.0, 90.0)
n_layers = st.sidebar.slider("Capas", 3, 7, 5)

uploaded_file = st.file_uploader("Cargar Datos (.tsv)", type=["tsv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep='\t')
    df.columns = [c.upper() for c in df.columns]
    
    # Limpieza forzada de tipos
    df[res_ch] = pd.to_numeric(df[res_ch], errors='coerce')
    df = df.dropna(subset=[res_ch, 'MD']).reset_index()

    engine = WFRD_Engine_Core()
    last_md, last_inc = df['MD'].iloc[-1], df['INC'].iloc[-1]

    with st.spinner('Mapeando Modelo Tierra...'):
        p, error = engine.solve(40, df[res_ch].values, df['MD'].values, last_inc, user_dip, n_layers)

    # Parámetros del modelo e interfaces
    res_vals, thick_vals = p[:n_layers], p[n_layers:2*n_layers-1]
    interfaces = np.cumsum(np.concatenate(([0], thick_vals))) - np.sum(thick_vals)/2

    # Proyección Adelante (Look-ahead)
    f_md, f_well, f_layer = engine.predict_ahead(last_md, last_inc, future_inc, user_dip)

    # --- MÉTRICAS DE MAPEO ---
    # DTBss en la broca
    curr_idx = np.searchsorted(interfaces, 0) - 1
    curr_idx = np.clip(curr_idx, 0, n_layers-2)
    
    st.markdown(f"### 🚩 Mapeo en Tiempo Real (Error: {error:.4f})")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("DTTB (TECHO)", f"{abs(interfaces[curr_idx]):.1f} ft")
    c2.metric("DTBB (BASE)", f"{abs(interfaces[curr_idx+1]):.1f} ft")
    c3.metric("NBI RELATIVO", f"{future_inc - user_dip:.1f}°")
    c4.success("MAPEO ACTIVO")

    # --- CURTAIN SECTION (MODELO TIERRA) ---
    tvd_grid = np.linspace(-60, 60, 150)
    md_total = np.concatenate([df['MD'].values, f_md])
    # Las capas se mueven con el DIP (Mapeo Dinámico)
    dip_offset = -(md_total - df['MD'].iloc[0]) * np.tan(np.radians(user_dip))
    
    z_map = np.zeros((len(tvd_grid), len(md_total)))
    for j in range(len(md_total)):
        shifted = interfaces + dip_offset[j]
        idx = np.searchsorted(shifted, tvd_grid)
        z_map[:, j] = res_vals[np.clip(idx, 0, n_layers-1)]

    fig = go.Figure()
    # Heatmap con Textura Tierra (Ruido sutil)
    fig.add_trace(go.Heatmap(z=np.log10(z_map) + np.random.normal(0,0.01,z_map.shape), 
                             x=md_total, y=tvd_grid, colorscale=color_theme, showscale=False))

    # Trayectoria Real + Proyección (Línea Blanca)
    well_path = np.concatenate([np.zeros(len(df)), f_well])
    fig.add_trace(go.Scatter(x=md_total, y=well_path, name="Wellbore", line=dict(color='white', width=4)))
    
    # Fronteras Geológicas (Mapeo)
    for inter in interfaces:
        fig.add_trace(go.Scatter(x=md_total, y=np.full(len(md_total), inter) + dip_offset, 
                                 mode='lines', line=dict(color='rgba(255,255,255,0.2)', width=1), showlegend=False))

    # Marcador de Broca y Labels
    fig.add_vline(x=last_md, line_dash="dash", line_color="white")
    fig.add_annotation(x=last_md, y=interfaces[curr_idx] + dip_offset[len(df)-1], text="TOP", font=dict(color="cyan"))
    fig.add_annotation(x=last_md, y=interfaces[curr_idx+1] + dip_offset[len(df)-1], text="BASE", font=dict(color="yellow"))

    fig.update_layout(height=600, template="plotly_dark", yaxis_title="TVD Relativo (ft)")
    st.plotly_chart(fig, use_container_width=True)

    # --- TRACK Q (TEMPLATE) ---
    fig_q = go.Figure()
    for q in q_chs:
        if q in df.columns: fig_q.add_trace(go.Scatter(x=df['MD'], y=df[q], name=q))
    fig_q.update_layout(height=250, template="plotly_dark", title="Validación con Canales Q")
    st.plotly_chart(fig_q, use_container_width=True)
