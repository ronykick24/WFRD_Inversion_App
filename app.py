import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from engine_wfrd import WFRD_Engine_Core

st.set_page_config(layout="wide", page_title="WFRD Geo-Mapper v4")

# --- SELECTOR DE TEMPLATE Y PALETA ---
st.sidebar.title("📋 Template de Control")
with st.sidebar.expander("Configuración de Canales", expanded=True):
    res_channel = st.selectbox("Curva de Inversión", ["AD2_GW6", "PD2_GW6", "AD4_GW6"])
    q_channels = st.multiselect("Visualizar Curvas Q", ["QPD2", "QPD4", "QPU1", "QPU4"], default=["QPD2", "QPU1"])
    color_map = st.selectbox("Paleta (Modelo Tierra)", ["Turbo", "Electric", "Viridis", "Hot"])

st.sidebar.title("🔭 Proyección y Mapeo")
user_dip = st.sidebar.slider("DIP de Capa (°)", -15.0, 15.0, 0.0)
n_layers = st.sidebar.slider("Número de Capas", 3, 7, 5)
proj_dist = st.sidebar.slider("Distancia Proyección (ft)", 0, 200, 100)

uploaded_file = st.file_uploader("Cargar Datos (.tsv)", type=["tsv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep='\t')
    df.columns = [c.upper() for c in df.columns]
    
    # Limpieza básica
    df = df.dropna(subset=[res_channel, 'MD', 'INC']).reset_index()

    engine = WFRD_Engine_Core()
    last_md = df['MD'].iloc[-1]
    last_inc = df['INC'].iloc[-1]

    # Ejecutar Inversión
    with st.spinner('Mapeando formación...'):
        p, error = engine.solve("Estocástico", 40, df[res_channel].values, df['MD'].values, last_inc, user_dip, n_layers)

    # Parámetros del modelo
    res_vals = p[:n_layers]
    thick_vals = p[n_layers:2*n_layers-1]
    interfaces = np.cumsum(np.concatenate(([0], thick_vals))) - np.sum(thick_vals)/2

    # Predicción Adelante
    f_md, f_well, f_layer_offset = engine.predict_ahead(last_md, last_inc, user_dip, proj_dist)

    # --- DASHBOARD DE MÉTRICAS ---
    # Encontrar en qué capa está el pozo actualmente para DTB
    current_layer = np.searchsorted(interfaces, 0) - 1
    current_layer = np.clip(current_layer, 0, n_layers-2)
    dt_top = abs(interfaces[current_layer])
    dt_base = abs(interfaces[current_layer+1])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("DTTB (TECHO)", f"{dt_top:.1f} ft")
    c2.metric("DTBB (BASE)", f"{dt_base:.1f} ft")
    c3.metric("DIP ACTUAL", f"{user_dip}°")
    c4.metric("ERROR INV.", f"{error:.4f}")

    # --- CORTINA DE MAPEO CON PROYECCIÓN ---
    st.subheader("🌐 Sección de Cortina: Mapeo y Proyección")
    
    tvd_grid = np.linspace(-60, 60, 150)
    md_total = np.concatenate([df['MD'].values, f_md])
    
    # El DIP genera el desplazamiento de las capas (Mapeo)
    dip_map_offset = -(md_total - df['MD'].iloc[0]) * np.tan(np.radians(user_dip))
    
    z_map = np.zeros((len(tvd_grid), len(md_total)))
    for j in range(len(md_total)):
        shifted_ints = interfaces + dip_map_offset[j]
        idx = np.searchsorted(shifted_ints, tvd_grid)
        z_map[:, j] = res_vals[np.clip(idx, 0, n_layers-1)]

    # Gráfico Principal
    fig = go.Figure()
    # Heatmap con textura tierra
    fig.add_trace(go.Heatmap(z=np.log10(z_map) + np.random.normal(0,0.01,z_map.shape), 
                             x=md_total, y=tvd_grid, colorscale=color_map, showscale=False))

    # Trayectoria (Real + Proyección)
    well_full_tvd = np.concatenate([np.zeros(len(df)), f_well])
    fig.add_trace(go.Scatter(x=md_total, y=well_full_tvd, name="Trayectoria", line=dict(color='white', width=4)))
    
    # Línea de Broca (Presente)
    fig.add_vline(x=last_md, line_dash="dash", line_color="white")

    # Mapeo de Líneas de Frontera
    for inter in interfaces:
        fig.add_trace(go.Scatter(x=md_total, y=np.full(len(md_total), inter) + dip_map_offset, 
                                 mode='lines', line=dict(color='rgba(255,255,255,0.2)', width=1), showlegend=False))

    # Anotaciones de DTB siguiendo la capa
    fig.add_annotation(x=last_md, y=interfaces[current_layer] + dip_map_offset[len(df)-1], 
                       text=f"TOP: {dt_top:.1f}ft", arrowhead=1, font=dict(color="white"))
    fig.add_annotation(x=last_md, y=interfaces[current_layer+1] + dip_map_offset[len(df)-1], 
                       text=f"BASE: {dt_base:.1f}ft", arrowhead=1, font=dict(color="white"))

    fig.update_layout(height=600, template="plotly_dark", yaxis_title="TVD Rel. (ft)")
    st.plotly_chart(fig, use_container_width=True)

    # --- REGISTRO Q (TEMPLATE SELECCIONADO) ---
    st.markdown("### 📈 Control de Geodirección (Curvas Q)")
    fig_q = go.Figure()
    for q in q_channels:
        if q in df.columns: fig_q.add_trace(go.Scatter(x=df['MD'], y=df[q], name=q))
    fig_q.update_layout(height=300, template="plotly_dark", xaxis_title="Measured Depth (ft)")
    st.plotly_chart(fig_q, use_container_width=True)
