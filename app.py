import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from engine_wfrd import WFRD_Engine_Core

st.set_page_config(layout="wide", page_title="WFRD Geo-Mapper V11")

# --- SIDEBAR ---
st.sidebar.title("🛠️ Configuración")
calc_mode = st.sidebar.selectbox("Inversión", ["Estocástico Global (1000 iters)", "Estocástico Local (100 iters)", "Determinístico"])
res_ch = st.sidebar.selectbox("Canal", ["AD2_GW6", "PD2_GW6", "AD4_GW6", "PU1_GW6"])
user_dip = st.sidebar.slider("DIP Capa (°)", -15.0, 15.0, 0.0)
sim_inc = st.sidebar.slider("Simular INC (°)", 80.0, 100.0, 90.0)
n_layers = st.sidebar.slider("Capas", 3, 7, 5)

uploaded_file = st.file_uploader("Cargar Datos (.tsv)", type=["tsv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep='\t')
    df.columns = [c.upper() for c in df.columns]
    
    # LIMPIEZA CRÍTICA DE DATOS (Evita TypeErrors)
    md_array = pd.to_numeric(df['MD'], errors='coerce').dropna().values
    res_array = pd.to_numeric(df[res_ch], errors='coerce').dropna().values
    min_len = min(len(md_array), len(res_array))
    md_array, res_array = md_array[:min_len], res_array[:min_len]
    
    last_md = float(md_array[-1])
    md_start = float(md_array[0])
    last_inc = float(df['INC'].dropna().iloc[-1])
    
    engine = WFRD_Engine_Core()

    with st.spinner('Ejecutando Inversión 3D...'):
        p, error = engine.solve(calc_mode, res_array, md_array, last_inc, user_dip, n_layers)

    res_h, thick, lambda_ani = p[:n_layers], p[n_layers:2*n_layers-1], p[-1]
    interfaces = np.cumsum(np.concatenate(([0], thick))) - np.sum(thick)/2

    # DTBss
    curr_idx = np.searchsorted(interfaces, 0) - 1
    curr_idx = np.clip(curr_idx, 0, n_layers-2)
    dttb, dtbb = abs(interfaces[curr_idx]), abs(interfaces[curr_idx+1])

    # MÉTRICAS
    st.markdown("### 🗺️ Dashboard de Navegación")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("↑ DTB Techo", f"{dttb:.1f} ft")
    m2.metric("↓ DTB Base", f"{dtbb:.1f} ft")
    m3.metric("Anisotropía (Rv/Rh)", f"{lambda_ani:.2f}")
    m4.metric("Ajuste (RMS)", f"{error:.4f}")

    # CORTINA GEOLÓGICA
    tvd_grid = np.linspace(-60, 60, 150)
    f_md = np.linspace(last_md, last_md + 200, 50)
    md_total = np.concatenate([md_array, f_md])
    
    # CORRECCIÓN DE DIP OFFSET (Uso de float puro md_start)
    dip_offset = -(md_total - md_start) * np.tan(np.radians(user_dip))
    well_path = np.concatenate([np.zeros(len(md_array)), (f_md - last_md) * np.sin(np.radians(sim_inc - 90))])

    z_map = np.zeros((len(tvd_grid), len(md_total)))
    for j in range(len(md_total)):
        shifted = interfaces + dip_offset[j]
        idx = np.searchsorted(shifted, tvd_grid)
        z_map[:, j] = res_h[np.clip(idx, 0, n_layers-1)]

    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=np.log10(z_map) + np.random.normal(0,0.01,z_map.shape), 
                             x=md_total, y=tvd_grid, colorscale="Turbo", showscale=False))

    # TRAYECTORIA Y LABELS DTBss
    fig.add_trace(go.Scatter(x=md_total, y=well_path, name="Trayectoria", line=dict(color='white', width=4)))
    
    # Flechas de DTBss en la posición actual
    idx_now = len(md_array) - 1
    fig.add_annotation(x=last_md, y=interfaces[curr_idx] + dip_offset[idx_now], text=f"↑ TOP: {dttb:.1f}ft", font=dict(color="cyan"))
    fig.add_annotation(x=last_md, y=interfaces[curr_idx+1] + dip_offset[idx_now], text=f"↓ BASE: {dtbb:.1f}ft", font=dict(color="yellow"))

    fig.update_layout(height=600, template="plotly_dark", yaxis_title="TVD Rel (ft)")
    st.plotly_chart(fig, use_container_width=True)
