import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from engine_wfrd import WFRD_Engine_Core

st.set_page_config(layout="wide", page_title="Geo-Mapper Pro 3D")

# --- SIDEBAR ---
st.sidebar.title("🛠️ Control de Geosteering")
calc_mode = st.sidebar.selectbox("Modo de Inversión", [
    "Estocástico Global (1000 iters)", 
    "Estocástico Local (100 iters)", 
    "Determinístico"
])

res_ch = st.sidebar.selectbox("Canal Inversión", ["AD2_GW6", "PD2_GW6", "AD4_GW6", "PU1_GW6"])
user_dip = st.sidebar.slider("DIP de Capa (°)", -15.0, 15.0, 0.0)
sim_inc = st.sidebar.slider("Simular Inclinación (°)", 80.0, 100.0, 90.0)
n_layers = st.sidebar.slider("Número de Capas", 3, 7, 5)

uploaded_file = st.file_uploader("Cargar Registro TSV", type=["tsv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep='\t')
    df.columns = [c.upper() for c in df.columns]
    engine = WFRD_Engine_Core()
    last_md, last_inc = df['MD'].iloc[-1], df['INC'].iloc[-1]

    with st.spinner(f'Calculando Inversión ({calc_mode})...'):
        p, error = engine.solve(calc_mode, df[res_ch], df['MD'], last_inc, user_dip, n_layers)

    res_h, thick, lambda_ani = p[:n_layers], p[n_layers:2*n_layers-1], p[-1]
    interfaces = np.cumsum(np.concatenate(([0], thick))) - np.sum(thick)/2

    # --- MÉTRICAS DTBss ---
    curr_idx = np.searchsorted(interfaces, 0) - 1
    curr_idx = np.clip(curr_idx, 0, n_layers-2)
    dttb, dtbb = abs(interfaces[curr_idx]), abs(interfaces[curr_idx+1])

    st.markdown("### 🗺️ Mapeo Proactivo y Análisis de Anisotropía")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("↑ DTB Techo", f"{dttb:.1f} ft")
    m2.metric("↓ DTB Base", f"{dtbb:.1f} ft")
    m3.metric("ANISOTROPÍA (Rv/Rh)", f"{lambda_ani:.2f}")
    m4.metric("ERROR RESIDUAL", f"{error:.4f}")

    # --- CORTINA GEOLÓGICA ---
    tvd_grid = np.linspace(-60, 60, 150)
    f_md = np.linspace(last_md, last_md + 200, 50)
    md_total = np.concatenate([df['MD'].values, f_md])
    
    # Mapeo: El DIP inclina las capas en el espacio
    dip_offset = -(md_total - df['MD'].iloc[0]) * np.tan(np.radians(user_dip))
    well_path = np.concatenate([np.zeros(len(df)), (f_md - last_md) * np.sin(np.radians(sim_inc - 90))])

    z_map = np.zeros((len(tvd_grid), len(md_total)))
    for j in range(len(md_total)):
        shifted = interfaces + dip_offset[j]
        idx = np.searchsorted(shifted, tvd_grid)
        z_map[:, j] = res_h[np.clip(idx, 0, n_layers-1)]

    fig = go.Figure()
    # Heatmap con textura de formación
    fig.add_trace(go.Heatmap(z=np.log10(z_map) + np.random.normal(0,0.01,z_map.shape), 
                             x=md_total, y=tvd_grid, colorscale="Turbo", showscale=False))

    # Trayectoria y Fronteras
    fig.add_trace(go.Scatter(x=md_total, y=well_path, name="Trayectoria", line=dict(color='white', width=4)))
    for inter in interfaces:
        fig.add_trace(go.Scatter(x=md_total, y=np.full(len(md_total), inter) + dip_offset, 
                                 mode='lines', line=dict(color='rgba(255,255,255,0.2)', width=1), showlegend=False))

    # LABELS DE DTBss DINÁMICOS
    fig.add_annotation(x=last_md, y=interfaces[curr_idx] + dip_offset[len(df)-1], 
                       text=f"↑ TOP: {dttb:.1f}ft", font=dict(color="cyan", size=12), arrowhead=2)
    fig.add_annotation(x=last_md, y=interfaces[curr_idx+1] + dip_offset[len(df)-1], 
                       text=f"↓ BASE: {dtbb:.1f}ft", font=dict(color="yellow", size=12), arrowhead=2)

    fig.update_layout(height=600, template="plotly_dark", yaxis_title="TVD Relativo (ft)")
    st.plotly_chart(fig, use_container_width=True)
