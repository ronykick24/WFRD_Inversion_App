import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from engine_wfrd import WFRD_Engine_Core

st.set_page_config(layout="wide", page_title="Geo-Mapper Ultimate")

# --- CONFIGURACIÓN DE TEMPLATE ---
st.sidebar.title("📋 Template de Curvas")
with st.sidebar.expander("Selección de Canales", expanded=True):
    res_ch = st.selectbox("Canal Resistividad", ["AD2_GW6", "PD2_GW6", "AD4_GW6"])
    q_chs = st.multiselect("Canales Q (Geodir)", ["QPD2", "QPD4", "QPU1", "QPU4"], default=["QPD2", "QPU1"])
    color_theme = st.selectbox("Textura Tierra", ["Turbo", "Electric", "Hot", "Cividis"])

# --- CONTROLES DE NAVEGACIÓN ---
st.sidebar.title("🔭 Geonavegación")
user_dip = st.sidebar.slider("DIP de Formación (°)", -15.0, 15.0, 0.0)
future_inc = st.sidebar.slider("Inclinación Proyectada (°)", 80.0, 100.0, 90.0)
n_layers = st.sidebar.slider("Número de Capas", 3, 7, 5)

uploaded_file = st.file_uploader("Cargar Datos (.tsv)", type=["tsv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep='\t')
    df.columns = [c.upper() for c in df.columns]
    df[res_ch] = pd.to_numeric(df[res_ch], errors='coerce')
    df = df.dropna(subset=[res_ch, 'MD']).reset_index()

    engine = WFRD_Engine_Core()
    last_md, last_inc = df['MD'].iloc[-1], df['INC'].iloc[-1]

    # Inversión
    with st.spinner('Mapeando Modelo Tierra...'):
        p, error = engine.solve(40, df[res_ch].values, df['MD'].values, last_inc, user_dip, n_layers)

    res_vals, thick_vals = p[:n_layers], p[n_layers:2*n_layers-1]
    interfaces = np.cumsum(np.concatenate(([0], thick_vals))) - np.sum(thick_vals)/2

    # --- LÓGICA DE MAPEO DINÁMICO ---
    curr_idx = np.searchsorted(interfaces, 0) - 1
    curr_idx = np.clip(curr_idx, 0, n_layers-2)
    dttb, dtbb = abs(interfaces[curr_idx]), abs(interfaces[curr_idx+1])

    # Predicción de Cruce
    exit_md, boundary = engine.predict_exit(last_md, dttb, dtbb, future_inc, user_dip)

    # --- DASHBOARD ---
    st.markdown(f"### 🛡️ Dashboard de Mapeo Proactivo")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("DTTB (TECHO)", f"{dttb:.1f} ft")
    c2.metric("DTBB (BASE)", f"{dtbb:.1f} ft")
    c3.metric("DIP / INC", f"{user_dip}° / {future_inc}°")
    
    if exit_md:
        c4.warning(f"⚠️ Cruce en {exit_md:.0f} ft ({boundary})")
    else:
        c4.success("✅ Trayectoria Paralela")

    # --- CURTAIN SECTION (MAPEO) ---
    tvd_grid = np.linspace(-60, 60, 150)
    # Proyección a 150ft
    f_md = np.linspace(last_md, last_md + 150, 40)
    md_total = np.concatenate([df['MD'].values, f_md])
    
    # Mapeo: El desplazamiento de capas sigue al DIP
    dip_offset = -(md_total - df['MD'].iloc[0]) * np.tan(np.radians(user_dip))
    
    # Trayectoria proyectada según inclinación manual
    f_well = (f_md - last_md) * np.sin(np.radians(future_inc - 90))
    well_path = np.concatenate([np.zeros(len(df)), f_well])

    z_map = np.zeros((len(tvd_grid), len(md_total)))
    for j in range(len(md_total)):
        shifted = interfaces + dip_offset[j]
        idx = np.searchsorted(shifted, tvd_grid)
        z_map[:, j] = res_vals[np.clip(idx, 0, n_layers-1)]

    fig = go.Figure()
    # Heatmap con Textura Tierra
    fig.add_trace(go.Heatmap(z=np.log10(z_map) + np.random.normal(0,0.01,z_map.shape), 
                             x=md_total, y=tvd_grid, colorscale=color_theme, showscale=False))

    # Trayectoria y Fronteras
    fig.add_trace(go.Scatter(x=md_total, y=well_path, name="Pozo", line=dict(color='white', width=4)))
    for inter in interfaces:
        fig.add_trace(go.Scatter(x=md_total, y=np.full(len(md_total), inter) + dip_offset, 
                                 mode='lines', line=dict(color='rgba(255,255,255,0.2)', width=1), showlegend=False))

    # Punto de Cruce Proyectado
    if exit_md:
        fig.add_trace(go.Scatter(x=[exit_md], y=[(exit_md-last_md)*np.sin(np.radians(future_inc-90))], 
                                 mode='markers', marker=dict(color='red', size=12, symbol='x'), name="Punto de Cruce"))

    fig.update_layout(height=600, template="plotly_dark", yaxis_title="TVD Rel. (ft)")
    st.plotly_chart(fig, use_container_width=True)

    # --- REGISTROS Q ---
    st.subheader("📉 Canales Q Seleccionados")
    fig_q = go.Figure()
    for q in q_chs:
        if q in df.columns: fig_q.add_trace(go.Scatter(x=df['MD'], y=df[q], name=q))
    fig_q.update_layout(height=250, template="plotly_dark")
    st.plotly_chart(fig_q, use_container_width=True)
