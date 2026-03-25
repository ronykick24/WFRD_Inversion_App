import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from engine_wfrd import WFRD_Engine_Core

st.set_page_config(layout="wide", page_title="Geo-Mapper Pro")

# --- 1. TEMPLATE DE CURVAS (SELECTOR) ---
st.sidebar.title("📋 Template de Log")
with st.sidebar.expander("Seleccionar Canales"):
    res_key = st.selectbox("Curva Principal (Inversión)", ["AD2_GW6", "AD4_GW6", "PD2_GW6", "PU1_GW6"])
    q_keys = st.multiselect("Curvas Geodirección (Q)", ["QPD2", "QPD4", "QPU1", "QPU4"], default=["QPD2", "QPU1"])
    color_theme = st.selectbox("Paleta de Tierra", ["Turbo", "Electric", "Hot", "Cividis"])

# --- 2. CONTROLES DE SIMULACIÓN ---
st.sidebar.title("🔭 Proyección & Dip")
user_dip = st.sidebar.slider("DIP Formación", -15.0, 15.0, 0.0)
proj_dist = st.sidebar.slider("Proyección adelante (ft)", 0, 200, 100)
n_layers = st.sidebar.slider("Capas", 3, 7, 5)

uploaded_file = st.file_uploader("Cargar TSV", type=["tsv"])

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file, sep='\t')
    df_raw.columns = [c.upper() for c in df_raw.columns]
    
    engine = WFRD_Engine_Core()
    last_md = df_raw['MD'].iloc[-1]
    last_inc = df_raw['INC'].iloc[-1]
    
    # Inversión
    p, misfit = engine.solve("Estocástico", 50, df_raw[res_key].values, df_raw['MD'].values, last_inc, user_dip, n_layers)
    
    # Geometría de capas
    res_vals = p[:n_layers]
    thick_vals = p[n_layers:2*n_layers-1]
    interfaces = np.cumsum(np.concatenate(([0], thick_vals))) - np.sum(thick_vals)/2

    # --- LÓGICA DE MAPEO (DTB SIGUE LA CAPA) ---
    # Calculamos la posición de la capa respecto a la broca (0,0)
    current_layer_idx = np.searchsorted(interfaces, 0) - 1
    current_layer_idx = np.clip(current_layer_idx, 0, n_layers-2)
    
    # Proyección futura
    f_md, f_path, f_layer = engine.predict_ahead(last_md, last_inc, user_dip, proj_dist)

    # --- VISUALIZACIÓN ---
    st.subheader(f"🌐 Mapeo Proactivo: Capas siguiendo DIP {user_dip}°")
    
    tvd_grid = np.linspace(-60, 60, 180)
    md_total = np.concatenate([df_raw['MD'].values, f_md])
    dip_offset = -(md_total - df_raw['MD'].iloc[0]) * np.tan(np.radians(user_dip))
    
    z_map = np.zeros((len(tvd_grid), len(md_total)))
    for j in range(len(md_total)):
        curr_ints = interfaces + dip_offset[j]
        idx = np.searchsorted(curr_ints, tvd_grid)
        z_map[:, j] = res_vals[np.clip(idx, 0, n_layers-1)]

    fig = go.Figure()
    # Heatmap con textura
    fig.add_trace(go.Heatmap(z=np.log10(z_map) + np.random.normal(0,0.01,z_map.shape), 
                             x=md_total, y=tvd_grid, colorscale=color_theme, showscale=False))

    # Trayectoria Real + Proyección (Línea Blanca)
    full_path = np.concatenate([np.zeros(len(df_raw)), f_path])
    fig.add_trace(go.Scatter(x=md_total, y=full_path, name="Trayectoria", line=dict(color='white', width=4)))
    
    # Línea de puntos para la broca
    fig.add_vline(x=last_md, line_dash="dash", line_color="rgba(255,255,255,0.5)")

    # MAPEO DE DTBss (Las líneas de capa siguen el Dip)
    for i, inter in enumerate(interfaces):
        fig.add_trace(go.Scatter(x=md_total, y=np.full(len(md_total), inter) + dip_offset, 
                                 mode='lines', line=dict(color='rgba(255,255,255,0.2)', width=1), showlegend=False))

    # Anotaciones DTB en la broca (Mapeo)
    d_top = interfaces[current_layer_idx] + dip_offset[len(df_raw)-1]
    d_base = interfaces[current_layer_idx+1] + dip_offset[len(df_raw)-1]
    
    fig.add_annotation(x=last_md, y=d_top, text=f"DTTB: {abs(d_top):.1f}ft", arrowhead=1, font=dict(color="cyan"))
    fig.add_annotation(x=last_md, y=d_base, text=f"DTBB: {abs(d_base):.1f}ft", arrowhead=1, font=dict(color="yellow"))

    fig.update_layout(height=600, template="plotly_dark", title="Mapeo de Formación con Proyección a 100ft")
    st.plotly_chart(fig, use_container_width=True)

    # Mostrar Curvas Q seleccionadas en el Template
    st.markdown("### 📈 Respuesta de Geodirección (Canales Seleccionados)")
    fig_q = go.Figure()
    for q in q_keys:
        if q in df_raw.columns: fig_q.add_trace(go.Scatter(x=df_raw['MD'], y=df_raw[q], name=q))
    fig_q.update_layout(height=250, template="plotly_dark")
    st.plotly_chart(fig_q, use_container_width=True)
