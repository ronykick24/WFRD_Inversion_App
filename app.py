import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils import clean_wfrd_data
from engine_wfrd import WFRD_Pro_Simulator

st.set_page_config(layout="wide", page_title="WFRD Geosteering Dashboard")

# Colores Geológicos con Shading Real (Agua-Aceite-Gas)
GEO_COLORS = [
    [0.0, 'rgb(0, 0, 139)'],    # Shale/Agua (Azul)
    [0.2, 'rgb(173, 216, 230)'], # Silt/Transición
    [0.4, 'rgb(240, 240, 240)'], # Caliza/Roca base
    [0.7, 'rgb(255, 215, 0)'],   # Arena Petrolífera (Oro)
    [1.0, 'rgb(139, 0, 0)']      # Gas/Alta Resistividad (Rojo)
]

st.sidebar.title("🎮 Panel de Control Geosteering")
inc_ctrl = st.sidebar.number_input("Inclinación de Broca (deg)", 0.0, 100.0, 85.0, step=0.1)
dip_ctrl = st.sidebar.slider("Ajuste de Dip (Buzamiento)", -15.0, 15.0, 0.0)
n_layers = st.sidebar.radio("Simulación de Capas", [3, 5], index=1)

uploaded_file = st.file_uploader("Cargar Registro WFRD (.tsv)", type=["tsv"])

if uploaded_file:
    df = clean_wfrd_data(pd.read_csv(uploaded_file, sep='\t'))
    sim = WFRD_Pro_Simulator()
    
    # Ejecutar Inversión con los parámetros del usuario
    with st.spinner('Actualizando Inversión 2D...'):
        p, misfit = sim.solve(df['AD2_GW6'].values, df['MD'].values, inc_ctrl, dip_ctrl)
    
    # --- CÁLCULO DE PROXIMIDAD ---
    # Interfaces reales en TVD
    thicknesses = p[5:9]
    interfaces = np.cumsum(np.concatenate(([0], thicknesses))) - np.sum(thicknesses)/2
    dist_to_top = abs(interfaces[2]) # Distancia al techo del reservorio (Capa 3)
    
    # --- ALERTAS Y MÉTRICAS ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Distancia al TECHO", f"{dist_to_top:.1f} ft")
    c2.metric("Misfit", f"{misfit:.4f}")
    
    if dist_to_top < 5.0:
        c3.error("⚠️ ALERTA: Proximidad al Techo (<5ft)")
    else:
        c3.success("✅ Trayectoria Segura")
    c4.metric("Resistividad Local", f"{p[2]:.1f} Ωm")

    # --- TRACKS HORIZONTALES (GR y RES) ---
    fig_tracks = go.Figure()
    # Simulación de Gamma Ray (GR) inverso a la resistividad
    gr_sim = 120 - (np.log10(df['AD2_GW6']) * 30) 
    fig_tracks.add_trace(go.Scatter(x=df['MD'], y=gr_sim, name="Gamma Ray", line=dict(color='green')))
    fig_tracks.add_trace(go.Scatter(x=df['MD'], y=df['AD2_GW6'], name="Resistividad", yaxis="y2", line=dict(color='red')))
    fig_tracks.update_layout(
        height=250, margin=dict(t=10, b=10),
        yaxis=dict(title="GR (API)", titlefont=dict(color="green")),
        yaxis2=dict(title="Res (Ωm)", titlefont=dict(color="red"), overlaying="y", side="right", type="log"),
        template="plotly_dark"
    )
    st.plotly_chart(fig_tracks, use_container_width=True)

    # --- CURTAIN SECTION CON SHADING REAL ---
    tvd_grid = np.linspace(-60, 60, 150)
    z_map = np.zeros((len(tvd_grid), len(df)))
    for i, z in enumerate(tvd_grid):
        l_idx = np.searchsorted(interfaces, z)
        z_map[i, :] = p[min(l_idx, 4)]

    fig_c = go.Figure()
    # Fondo con degradado geológico
    fig_c.add_trace(go.Heatmap(
        z=np.log10(z_map), x=df['MD'], y=tvd_grid,
        colorscale=GEO_COLORS, zsmooth='best', colorbar=dict(title="Log Res")
    ))

    # Trayectoria dinámica cruzando capas
    rel_angle = np.radians(inc_ctrl - 90 - dip_ctrl)
    well_path = (df['MD'] - df['MD'].min()) * np.tan(rel_angle)
    fig_c.add_trace(go.Scatter(x=df['MD'], y=well_path, name="Broca", line=dict(color='white', width=5)))

    # Lables de Distancia en el gráfico
    fig_c.add_annotation(x=df['MD'].iloc[-1], y=well_path[-1]+5, text=f"Top: {dist_to_top:.1f}ft", showarrow=False, font=dict(color="white"))

    fig_c.update_layout(height=550, title="Sección Estructural Interactiva (Visualización 50ft)", 
                        yaxis=dict(title="TVD Relativo (ft)"), template="plotly_dark")
    st.plotly_chart(fig_c, use_container_width=True)
