import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils import clean_wfrd_data
from engine_wfrd import WFRD_Pro_Engine

st.set_page_config(layout="wide", page_title="WFRD GuideWave Pro")

# Paleta Geológica Pro
GEO_COLORS = [[0, '#00008B'], [0.3, '#87CEEB'], [0.5, '#F0F0F0'], [0.8, '#FFD700'], [1, '#8B0000']]

# --- SIDEBAR INTERACTIVO ---
st.sidebar.title("🛠️ Configuración Geosteering")
n_layers = st.sidebar.select_slider("Cantidad de Capas", options=[2, 3, 4, 5], value=5)
nbi_target = st.sidebar.slider("NBI Target (Proximidad)", 0.0, 1.0, 0.8)
fixed_thick = st.sidebar.number_input("Espesor de Capa Base (ft)", 5.0, 50.0, 15.0)

# Datos de Trayectoria (Labels en tiempo real)
c_inc, c_azm, c_misfit = st.columns(3)

uploaded_file = st.file_uploader("Cargar Registro TSV", type=["tsv"])

if uploaded_file:
    df = clean_wfrd_data(pd.read_csv(uploaded_file, sep='\t'))
    engine = WFRD_Pro_Engine()
    
    # Inputs dinámicos
    last_inc = df['INC'].iloc[-1]
    last_azm = df['AZM'].iloc[-1] if 'AZM' in df.columns else 0.0
    
    c_inc.metric("Última Inclinación", f"{last_inc:.2f}°")
    c_azm.metric("Último Azimut", f"{last_azm:.2f}°")

    user_dip = st.slider("Ajustar Buzamiento (Dip) de la Formación", -15.0, 15.0, 0.0)

    # Inversión
    with st.spinner('Sincronizando capas con Dip...'):
        p, misfit = engine.solve(df['AD2_GW6'].values, df['MD'].values, last_inc, user_dip, n_layers)
    
    c_misfit.metric("Misfit (Validación)", f"{misfit:.4f}")

    # --- TRACKS HORIZONTALES (RESISTIVIDAD Y GR) ---
    fig_h = go.Figure()
    fig_h.add_trace(go.Scatter(x=df['MD'], y=df['AD2_GW6'], name="Resistividad (33ft)", line=dict(color='red')))
    fig_h.add_trace(go.Scatter(x=df['MD'], y=df['AU1_GW6'], name="Resistividad (50ft)", line=dict(color='orange')))
    fig_h.update_layout(height=250, yaxis_type="log", title="Registros Horizontales", template="plotly_dark")
    st.plotly_chart(fig_h, use_container_width=True)

    # --- CURTAIN SECTION CON MOVIMIENTO DE DIP ---
    # Creamos una malla TVD y aplicamos la rotación del Dip
    tvd_grid = np.linspace(-60, 60, 120)
    thicknesses = p[5:9]
    interfaces = np.cumsum(np.concatenate(([0], thicknesses))) - np.sum(thicknesses)/2
    
    # El truco para que las capas se muevan: El centro de la capa cambia con el MD y el Dip
    z_map = np.zeros((len(tvd_grid), len(df)))
    dip_offset = (df['MD'].values - df['MD'].values[0]) * np.tan(np.radians(user_dip))
    
    for j, md_val in enumerate(df['MD']):
        current_interfaces = interfaces + dip_offset[j]
        for i, z_val in enumerate(tvd_grid):
            idx = np.searchsorted(current_interfaces, z_val)
            z_map[i, j] = p[min(idx, 4)]

    fig_c = go.Figure()
    fig_c.add_trace(go.Heatmap(
        z=np.log10(z_map), x=df['MD'], y=tvd_grid,
        colorscale=GEO_COLORS, zsmooth='best'
    ))

    # Trayectoria del pozo (Línea central)
    fig_c.add_trace(go.Scatter(x=df['MD'], y=np.zeros(len(df)), name="Trayectoria Pozo", line=dict(color='black', width=4)))
    
    fig_c.update_layout(height=600, title=f"Sección de Cortina con Dip de {user_dip}°", 
                        yaxis_title="TVD Relativo (ft)", template="plotly_white")
    st.plotly_chart(fig_c, use_container_width=True)
    
    st.warning(f"Distancia estimada al techo del reservorio: {abs(interfaces[2]):.1f} ft (NBI: {nbi_target})")
