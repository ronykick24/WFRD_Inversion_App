import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from engine_wfrd import WFRD_Engine_Core

st.set_page_config(layout="wide", page_title="Geo-Mapper Ultimate v12")

# --- SIDEBAR ---
st.sidebar.title("🛠️ Control de Geosteering")
calc_mode = st.sidebar.selectbox("Inversión Seleccionada", [
    "Estocástico Global (1000 iters)", 
    "Estocástico Local (100 iters)", 
    "Determinístico"
])

res_ch = st.sidebar.selectbox("Canal Resistividad", ["AD2_GW6", "PD2_GW6", "AD4_GW6", "PU1_GW6"])
user_dip = st.sidebar.slider("DIP Formación (°)", -15.0, 15.0, 0.0)
sim_inc = st.sidebar.slider("Simular INC (°)", 80.0, 100.0, 90.0)
n_layers = st.sidebar.slider("Capas en Modelo", 3, 7, 5)

uploaded_file = st.file_uploader("Cargar Datos (.tsv)", type=["tsv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep='\t')
    df.columns = [c.upper() for c in df.columns]
    
    # Limpieza de datos asegurando escalares float
    md_clean = pd.to_numeric(df['MD'], errors='coerce').dropna()
    res_clean = pd.to_numeric(df[res_ch], errors='coerce').loc[md_clean.index]
    
    md_array = md_clean.values
    res_array = res_clean.values
    
    last_md = float(md_array[-1])
    md_start = float(md_array[0])
    last_inc = float(df['INC'].dropna().iloc[-1])
    
    engine = WFRD_Engine_Core()

    with st.spinner(f'Calculando Inversión ({calc_mode})...'):
        p, error = engine.solve(calc_mode, res_array, md_array, last_inc, user_dip, n_layers)

    res_h, thick, lambda_ani = p[:n_layers], p[n_layers:2*n_layers-1], p[-1]
    interfaces = np.cumsum(np.concatenate(([0], thick))) - np.sum(thick)/2

    # Cálculos DTBss
    curr_idx = np.searchsorted(interfaces, 0) - 1
    curr_idx = np.clip(curr_idx, 0, n_layers-2)
    dttb, dtbb = abs(interfaces[curr_idx]), abs(interfaces[curr_idx+1])

    # DASHBOARD
    st.markdown("### 🗺️ Mapeo Geológico y Proyección de Trayectoria")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("↑ DTB Techo", f"{dttb:.1f} ft")
    c2.metric("↓ DTB Base", f"{dtbb:.1f} ft")
    c3.metric("ANISOTROPÍA (Rv/Rh)", f"{lambda_ani:.2f}")
    c4.metric("ERROR MODELO", f"{error:.4f}")

    # CORTINA GEOLÓGICA PROACTIVA
    tvd_grid = np.linspace(-60, 60, 150)
    f_md = np.linspace(last_md, last_md + 200, 50)
    md_total = np.concatenate([md_array, f_md])
    
    dip_offset = -(md_total - md_start) * np.tan(np.radians(user_dip))
    well_path = np.concatenate([np.zeros(len(md_array)), (f_md - last_md) * np.sin(np.radians(sim_inc - 90))])

    z_map = np.zeros((len(tvd_grid), len(md_total)))
    for j in range(len(md_total)):
        shifted = interfaces + dip_offset[j]
        idx = np.searchsorted(shifted, tvd_grid)
        z_map[:, j] = res_h[np.clip(idx, 0, n_layers-1)]

    fig = go.Figure()
    # Heatmap con textura tierra
    fig.add_trace(go.Heatmap(z=np.log10(z_map) + np.random.normal(0,0.012,z_map.shape), 
                             x=md_total, y=tvd_grid, colorscale="Turbo", showscale=False))

    # Trayectoria y Labels dinámicos
    fig.add_trace(go.Scatter(x=md_total, y=well_path, name="Trayectoria", line=dict(color='white', width=4)))
    
    idx_now = len(md_array) - 1
    fig.add_annotation(x=last_md, y=interfaces[curr_idx] + dip_offset[idx_now], 
                       text=f"↑ TOP: {dttb:.1f}ft", font=dict(color="cyan", size=12), arrowhead=2)
    fig.add_annotation(x=last_md, y=interfaces[curr_idx+1] + dip_offset[idx_now], 
                       text=f"↓ BASE: {dtbb:.1f}ft", font=dict(color="yellow", size=12), arrowhead=2)

    fig.update_layout(height=600, template="plotly_dark", yaxis_title="TVD Relativo (ft)")
    st.plotly_chart(fig, use_container_width=True)
