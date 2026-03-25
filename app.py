import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from engine_wfrd import WFRD_Engine_Core

st.set_page_config(layout="wide", page_title="WFRD Geo-Mapper Ultimate")

# --- SIDEBAR ---
st.sidebar.title("🛠️ Configuración Motor")
calc_mode = st.sidebar.selectbox("Algoritmo", ["Estocástico Global (1000 iters)", "Estocástico Local (100 iters)", "Determinístico"])
res_ch = st.sidebar.selectbox("Canal Inversión", ["AD2_GW6", "PD2_GW6", "AD4_GW6", "PU1_GW6"])
user_dip = st.sidebar.slider("DIP Formación (°)", -15.0, 15.0, 0.0)
sim_inc = st.sidebar.slider("Simular INC (°)", 80.0, 100.0, 90.0)
n_layers = st.sidebar.slider("Capas", 3, 7, 5)

uploaded_file = st.file_uploader("Cargar Registro TSV", type=["tsv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep='\t')
    df.columns = [c.upper() for c in df.columns]
    
    # BLINDAJE DE VALORES ESCALARES (Resuelve el TypeError)
    last_md = float(df['MD'].dropna().iloc[-1])
    last_inc = float(df['INC'].dropna().iloc[-1])
    
    engine = WFRD_Engine_Core()

    with st.spinner(f'Calculando Inversión 3D...'):
        p, error = engine.solve(calc_mode, df[res_ch], df['MD'], last_inc, user_dip, n_layers)

    res_h, thick, lambda_ani = p[:n_layers], p[n_layers:2*n_layers-1], p[-1]
    interfaces = np.cumsum(np.concatenate(([0], thick))) - np.sum(thick)/2

    # DTBss Geométrico
    curr_idx = np.searchsorted(interfaces, 0) - 1
    curr_idx = np.clip(curr_idx, 0, n_layers-2)
    dttb, dtbb = abs(interfaces[curr_idx]), abs(interfaces[curr_idx+1])

    # DASHBOARD
    st.markdown("### 🗺️ Mapeo Geológico Proactivo")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("↑ DTB Techo", f"{dttb:.1f} ft")
    m2.metric("↓ DTB Base", f"{dtbb:.1f} ft")
    m3.metric("ANISOTROPÍA (Rv/Rh)", f"{lambda_ani:.2f}")
    m4.metric("ERROR", f"{error:.4f}")

    # CORTINA CON MAPEO Y TEXTURA
    tvd_grid = np.linspace(-60, 60, 150)
    # Proyección segura con MD escalar
    f_md = np.linspace(last_md, last_md + 200, 50) 
    md_total = np.concatenate([df['MD'].values, f_md])
    
    # Mapeo: DIP afecta la posición de las capas
    dip_offset = -(md_total - df['MD'].iloc[0]) * np.tan(np.radians(user_dip))
    well_path = np.concatenate([np.zeros(len(df)), (f_md - last_md) * np.sin(np.radians(sim_inc - 90))])

    z_map = np.zeros((len(tvd_grid), len(md_total)))
    for j in range(len(md_total)):
        shifted = interfaces + dip_offset[j]
        idx = np.searchsorted(shifted, tvd_grid)
        z_map[:, j] = res_h[np.clip(idx, 0, n_layers-1)]

    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=np.log10(z_map) + np.random.normal(0,0.012,z_map.shape), 
                             x=md_total, y=tvd_grid, colorscale="Turbo", showscale=False))

    # Fronteras, Trayectoria y LABELS
    fig.add_trace(go.Scatter(x=md_total, y=well_path, name="Pozo", line=dict(color='white', width=4)))
    
    # LABELS DTBss EN LA BROCA (BIT)
    fig.add_annotation(x=last_md, y=interfaces[curr_idx] + dip_offset[len(df)-1], 
                       text=f"↑ TOP: {dttb:.1f}ft", font=dict(color="cyan", size=12), arrowhead=2)
    fig.add_annotation(x=last_md, y=interfaces[curr_idx+1] + dip_offset[len(df)-1], 
                       text=f"↓ BASE: {dtbb:.1f}ft", font=dict(color="yellow", size=12), arrowhead=2)

    fig.update_layout(height=650, template="plotly_dark", yaxis_title="TVD Relativo (ft)")
    st.plotly_chart(fig, use_container_width=True)
