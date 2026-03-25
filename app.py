import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from engine_wfrd import WFRD_Engine_Core

st.set_page_config(layout="wide", page_title="WFRD Geo-Mapper Pro")

# --- SIDEBAR: CONTROL TOTAL ---
st.sidebar.title("🎮 Parámetros de Control")
calc_mode = st.sidebar.selectbox("Inversión y Esfuerzo", [
    "Estocástico Global (1000 iters)", 
    "Estocástico Local (100 iters)", 
    "Determinístico (Fast)"
])

with st.sidebar.expander("Modelo de Tierra & Anisotropía", expanded=True):
    res_ch = st.selectbox("Canal Resistividad", ["AD2_GW6", "PD2_GW6", "AD4_GW6", "PU1_GW6"])
    n_layers = st.slider("Número de Capas", 3, 9, 5)
    color_theme = st.selectbox("Paleta Visual", ["Turbo", "Electric", "Hot", "Cividis"])

st.sidebar.title("🔭 Geonavegación 3D Rel.")
user_dip = st.sidebar.slider("DIP de Capa (°)", -15.0, 15.0, 0.0)
sim_inc = st.sidebar.slider("Proyectar Inclinación (°)", 80.0, 100.0, 90.0)

uploaded_file = st.file_uploader("Cargar Datos (.tsv)", type=["tsv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep='\t')
    df.columns = [c.upper() for c in df.columns]
    engine = WFRD_Engine_Core()
    last_md, last_inc = df['MD'].iloc[-1], df['INC'].iloc[-1]

    # Ejecución de Inversión
    with st.spinner(f'Calculando Inversión con Anisotropía...'):
        p, error = engine.solve(calc_mode, df[res_ch], df['MD'], last_inc, user_dip, n_layers)

    # Desempaque de resultados (Incluye Lambda al final)
    res_vals, thick_vals, lambda_ani = p[:n_layers], p[n_layers:2*n_layers-1], p[-1]
    interfaces = np.cumsum(np.concatenate(([0], thick_vals))) - np.sum(thick_vals)/2

    # --- LÓGICA DE DTB DINÁMICA ---
    curr_idx = np.searchsorted(interfaces, 0) - 1
    curr_idx = np.clip(curr_idx, 0, n_layers-2)
    dttb, dtbb = abs(interfaces[curr_idx]), abs(interfaces[curr_idx+1])
    exit_md, target = engine.predict_exit(last_md, dttb, dtbb, sim_inc, user_dip)

    # --- TRACKS DE LOGS ---
    t_fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=("Resistividad Invertida", "Validación Q"))
    t_fig.add_trace(go.Scatter(x=df[res_ch], y=df['MD'], name="Observed", line=dict(color='lime')), row=1, col=1)
    t_fig.update_yaxes(autorange="reversed")
    t_fig.update_xaxes(type="log", col=1)
    t_fig.update_layout(height=300, template="plotly_dark")
    st.plotly_chart(t_fig, use_container_width=True)

    # --- DASHBOARD MÉTRICO ---
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("↑ DTBss TECHO", f"{dttb:.1f} ft")
    c2.metric("↓ DTBss BASE", f"{dtbb:.1f} ft")
    c3.metric("ANISOTROPÍA (Rv/Rh)", f"{lambda_ani:.2f}")
    c4.metric("ERROR MODELO", f"{error:.4f}")

    # --- CORTINA DE MAPEO (MODELO 2D/3D RELATIVO) ---
    tvd_grid = np.linspace(-60, 60, 150)
    f_md = np.linspace(last_md, last_md + 200, 50)
    md_total = np.concatenate([df['MD'].values, f_md])
    
    # Mapeo: El DIP inclina todo el sistema de coordenadas
    dip_offset = -(md_total - df['MD'].iloc[0]) * np.tan(np.radians(user_dip))
    f_well = (f_md - last_md) * np.sin(np.radians(sim_inc - 90))
    well_path = np.concatenate([np.zeros(len(df)), f_well])

    z_map = np.zeros((len(tvd_grid), len(md_total)))
    for j in range(len(md_total)):
        shifted = interfaces + dip_offset[j]
        idx = np.searchsorted(shifted, tvd_grid)
        z_map[:, j] = res_vals[np.clip(idx, 0, n_layers-1)]

    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=np.log10(z_map) + np.random.normal(0, 0.01, z_map.shape), 
                             x=md_total, y=tvd_grid, colorscale=color_theme, showscale=False))

    # Fronteras y Labels DTBss
    for inter in interfaces:
        fig.add_trace(go.Scatter(x=md_total, y=np.full(len(md_total), inter) + dip_offset, 
                                 mode='lines', line=dict(color='rgba(255,255,255,0.2)', width=1), showlegend=False))

    fig.add_trace(go.Scatter(x=md_total, y=well_path, name="Trayectoria", line=dict(color='white', width=4)))

    # LABELS DTBss REALES EN EL BIT
    fig.add_annotation(x=last_md, y=interfaces[curr_idx] + dip_offset[len(df)-1], 
                       text=f"↑ DTB: {dttb:.1f}ft", arrowhead=2, font=dict(color="cyan"))
    fig.add_annotation(x=last_md, y=interfaces[curr_idx+1] + dip_offset[len(df)-1], 
                       text=f"↓ DTB: {dtbb:.1f}ft", arrowhead=2, font=dict(color="yellow"))

    if exit_md and exit_md < md_total[-1]:
        y_ex = (exit_md - last_md) * np.sin(np.radians(sim_inc - 90))
        fig.add_trace(go.Scatter(x=[exit_md], y=[y_ex], mode='markers', marker=dict(color='red', size=15, symbol='x'), name="Cruce"))

    fig.update_layout(height=600, template="plotly_dark", yaxis_title="TVD Relativo (ft)")
    st.plotly_chart(fig, use_container_width=True)
