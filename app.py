import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from engine_wfrd import WFRD_Engine_Core

st.set_page_config(layout="wide", page_title="Geo-Mapper Ultimate v7")

# --- 1. TEMPLATE Y CONFIGURACIÓN ---
st.sidebar.title("🎨 Template & Visual")
with st.sidebar.expander("Selección de Canales y Colores", expanded=True):
    res_ch = st.selectbox("Curva Inversión", ["AD2_GW6", "PD2_GW6", "AD4_GW6", "PU1_GW6"])
    q_chs = st.multiselect("Canales Q", ["QPD2", "QPD4", "QPU1", "QPU4"], default=["QPD2", "QPU1"])
    color_theme = st.selectbox("Paleta Modelo Tierra", ["Turbo", "Electric", "Hot", "Cividis", "Viridis"])

st.sidebar.title("🔭 Geonavegación Proactiva")
user_dip = st.sidebar.slider("DIP Formación (°)", -15.0, 15.0, 0.0)
future_inc = st.sidebar.slider("Simular Inclinación (°)", 80.0, 100.0, 90.0)
n_layers = st.sidebar.slider("Cantidad de Capas", 3, 9, 5)

uploaded_file = st.file_uploader("Cargar Registro TSV", type=["tsv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep='\t')
    df.columns = [c.upper() for c in df.columns]
    
    engine = WFRD_Engine_Core()
    last_md, last_inc = df['MD'].iloc[-1], df['INC'].iloc[-1]

    # Inversión
    with st.spinner('Mapeando Formación...'):
        p, error = engine.solve(40, df[res_ch], df['MD'], last_inc, user_dip, n_layers)

    res_vals, thick_vals = p[:n_layers], p[n_layers:2*n_layers-1]
    interfaces = np.cumsum(np.concatenate(([0], thick_vals))) - np.sum(thick_vals)/2

    # Lógica de Mapeo Dinámico (DTBss sigue la capa)
    curr_idx = np.searchsorted(interfaces, 0) - 1
    curr_idx = np.clip(curr_idx, 0, n_layers-2)
    dttb, dtbb = abs(interfaces[curr_idx]), abs(interfaces[curr_idx+1])
    
    # Punto de Salida
    exit_md, target_name = engine.predict_exit(last_md, dttb, dtbb, future_inc, user_dip)

    # --- TRACKS VERTICALES (LOGS) ---
    st.subheader("📉 Registros en Tiempo Real")
    fig_logs = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=("Resistividad", "Geodirección (Q)"))
    
    fig_logs.add_trace(go.Scatter(x=df[res_ch], y=df['MD'], name=res_ch, line=dict(color='lime')), row=1, col=1)
    for q in q_chs:
        if q in df.columns: fig_logs.add_trace(go.Scatter(x=df[q], y=df['MD'], name=q), row=1, col=2)
    
    fig_logs.update_yaxes(autorange="reversed")
    fig_logs.update_xaxes(type="log", col=1)
    fig_logs.update_layout(height=400, template="plotly_dark")
    st.plotly_chart(fig_logs, use_container_width=True)

    # --- DASHBOARD DE MÉTRICAS ---
    st.markdown("### 🗺️ Mapeo Proactivo y DTBss")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("DTBss TECHO (↑)", f"{dttb:.1f} ft")
    m2.metric("DTBss BASE (↓)", f"{dtbb:.1f} ft")
    m3.metric("ESPESOR CAPA", f"{thick_vals[curr_idx]:.1f} ft")
    
    if exit_md and (exit_md - last_md) < 150:
        m4.error(f"⚠️ CRUCE EN {int(exit_md)} ft ({target_name})")
    else:
        m4.success("✅ TRAYECTORIA EN VENTANA")

    # --- CURTAIN SECTION (MODELO TIERRA CON TEXTURA) ---
    tvd_grid = np.linspace(-60, 60, 150)
    f_md = np.linspace(last_md, last_md + 150, 40)
    md_total = np.concatenate([df['MD'].values, f_md])
    
    # Mapeo: Desplazamiento de capas por DIP
    dip_offset = -(md_total - df['MD'].iloc[0]) * np.tan(np.radians(user_dip))
    
    # Proyección Trayectoria
    f_well = (f_md - last_md) * np.sin(np.radians(future_inc - 90))
    well_path = np.concatenate([np.zeros(len(df)), f_well])

    z_map = np.zeros((len(tvd_grid), len(md_total)))
    for j in range(len(md_total)):
        shifted = interfaces + dip_offset[j]
        idx = np.searchsorted(shifted, tvd_grid)
        z_map[:, j] = res_vals[np.clip(idx, 0, n_layers-1)]

    fig_curtain = go.Figure()
    # Heatmap con textura de roca
    fig_curtain.add_trace(go.Heatmap(z=np.log10(z_map) + np.random.normal(0, 0.012, z_map.shape), 
                                   x=md_total, y=tvd_grid, colorscale=color_theme, showscale=False))

    # Fronteras y Trayectoria
    fig_curtain.add_trace(go.Scatter(x=md_total, y=well_path, name="Pozo", line=dict(color='white', width=4)))
    for inter in interfaces:
        fig_curtain.add_trace(go.Scatter(x=md_total, y=np.full(len(md_total), inter) + dip_offset, 
                                        mode='lines', line=dict(color='rgba(255,255,255,0.3)', width=1), showlegend=False))

    # LABELS DE DTBss (ARRIBA Y ABAJO)
    fig_curtain.add_annotation(x=last_md, y=interfaces[curr_idx] + dip_offset[len(df)-1], 
                              text=f"↑ TOP: {dttb:.1f} ft", showarrow=True, arrowhead=2, font=dict(color="cyan", size=12))
    fig_curtain.add_annotation(x=last_md, y=interfaces[curr_idx+1] + dip_offset[len(df)-1], 
                              text=f"↓ BASE: {dtbb:.1f} ft", showarrow=True, arrowhead=2, font=dict(color="yellow", size=12))

    # Marcador de Salida Futura
    if exit_md and exit_md < md_total[-1]:
        y_exit = (exit_md - last_md) * np.sin(np.radians(future_inc - 90))
        fig_curtain.add_trace(go.Scatter(x=[exit_md], y=[y_exit], mode='markers', 
                                       marker=dict(color='red', size=15, symbol='x'), name="Punto de Salida"))

    fig_curtain.update_layout(height=650, template="plotly_dark", title="Sección de Mapeo Proactivo (Modelo Tierra)", yaxis_title="TVD Relativo (ft)")
    st.plotly_chart(fig_curtain, use_container_width=True)
