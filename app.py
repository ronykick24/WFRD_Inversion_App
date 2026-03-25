import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from engine_wfrd import WFRD_Engine_Core

st.set_page_config(layout="wide", page_title="Geo-Mapper Pro V6")

# --- SIDEBAR: TEMPLATE Y GEONAVEGACIÓN ---
st.sidebar.title("📋 Configuración")
with st.sidebar.expander("Template de Curvas", expanded=True):
    res_ch = st.selectbox("Curva Inversión", ["AD2_GW6", "PD2_GW6", "AD4_GW6"])
    q_chs = st.multiselect("Curvas Q", ["QPD2", "QPD4", "QPU1", "QPU4"], default=["QPD2", "QPU1"])
    color_theme = st.selectbox("Paleta Tierra", ["Turbo", "Electric", "Hot", "Cividis"])

st.sidebar.title("🔭 Proyección")
user_dip = st.sidebar.slider("DIP Formación (°)", -15.0, 15.0, 0.0)
future_inc = st.sidebar.slider("Inclinación Futura (°)", 80.0, 100.0, 90.0)
n_layers = st.sidebar.slider("Capas", 3, 7, 5)

uploaded_file = st.file_uploader("Cargar Datos (.tsv)", type=["tsv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep='\t')
    df.columns = [c.upper() for c in df.columns]
    
    engine = WFRD_Engine_Core()
    last_md, last_inc = df['MD'].iloc[-1], df['INC'].iloc[-1]

    with st.spinner('Calculando Mapeo...'):
        p, error = engine.solve(40, df[res_ch].values, df['MD'].values, last_inc, user_dip, n_layers)

    res_vals, thick_vals = p[:n_layers], p[n_layers:2*n_layers-1]
    interfaces = np.cumsum(np.concatenate(([0], thick_vals))) - np.sum(thick_vals)/2

    # Mapeo de posición actual
    curr_idx = np.searchsorted(interfaces, 0) - 1
    curr_idx = np.clip(curr_idx, 0, n_layers-2)
    dttb, dtbb = abs(interfaces[curr_idx]), abs(interfaces[curr_idx+1])

    # Cálculo de Cruce Proyectado
    exit_md, target_name = engine.predict_exit(last_md, dttb, dtbb, future_inc, user_dip)

    # --- MÉTRICAS ---
    st.markdown("### 🗺️ Estado del Mapeo Proactivo")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("DTTB (TECHO)", f"{dttb:.1f} ft")
    m2.metric("DTBB (BASE)", f"{dtbb:.1f} ft")
    m3.metric("DIST. CRUCE", f"{int(exit_md - last_md) if exit_md else 0} ft")
    
    if exit_md and (exit_md - last_md) < 100:
        m4.error(f"⚠️ CRUCE EN {boundary}") if 'boundary' in locals() else m4.error(f"⚠️ CRUCE: {target_name}")
    else:
        m4.success("✅ TRAYECTORIA SEGURA")

    # --- SECCIÓN DE CORTINA ---
    tvd_grid = np.linspace(-60, 60, 150)
    f_md = np.linspace(last_md, last_md + 150, 40)
    md_total = np.concatenate([df['MD'].values, f_md])
    
    # El DIP mapea las capas (desplazamiento vertical)
    dip_offset = -(md_total - df['MD'].iloc[0]) * np.tan(np.radians(user_dip))
    
    # Trayectoria proyectada (TVD relativo)
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

    # Trayectoria y Fronteras de Mapeo
    fig.add_trace(go.Scatter(x=md_total, y=well_path, name="Trayectoria", line=dict(color='white', width=4)))
    for inter in interfaces:
        fig.add_trace(go.Scatter(x=md_total, y=np.full(len(md_total), inter) + dip_offset, 
                                 mode='lines', line=dict(color='rgba(255,255,255,0.2)', width=1), showlegend=False))

    # Marcador de Cruce Proyectado
    if exit_md and exit_md < md_total[-1]:
        y_exit = (exit_md - last_md) * np.sin(np.radians(future_inc - 90))
        fig.add_trace(go.Scatter(x=[exit_md], y=[y_exit], mode='markers', 
                                 marker=dict(color='red', size=15, symbol='x'), name="Punto de Salida"))

    fig.update_layout(height=600, template="plotly_dark", yaxis_title="TVD Relativo (ft)")
    st.plotly_chart(fig, use_container_width=True)
