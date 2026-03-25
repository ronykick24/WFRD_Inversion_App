import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils import clean_wfrd_data
from engine_wfrd import WFRD_Advanced_Engine

st.set_page_config(layout="wide", page_title="WFRD Geosteering AI")

# Paleta de Alto Contraste: Azul (Arcilla) -> Gris -> Amarillo (Aceite) -> Rojo (Gas)
GEO_COLORS = [
    [0.0, '#000080'], [0.2, '#87CEEB'], [0.4, '#FFFFFF'], 
    [0.7, '#FFD700'], [1.0, '#8B0000']
]

st.sidebar.title("🎮 Panel de Control")
user_inc = st.sidebar.slider("Inclinación de Broca (°)", 75.0, 95.0, 85.0)
user_dip = st.sidebar.slider("Buzamiento (Dip)", -10.0, 10.0, 0.0)

uploaded_file = st.file_uploader("Cargar Registro TSV", type=["tsv"])

if uploaded_file:
    df = clean_wfrd_data(pd.read_csv(uploaded_file, sep='\t'))
    engine = WFRD_Advanced_Engine()
    
    with st.spinner('Simulando capas estocásticas...'):
        p, misfit = engine.solve(df['AD2_GW6'].values, df['MD'].values, user_inc, user_dip)
    
    # --- EXTRACCIÓN DE INTERFACES ---
    thicknesses = p[5:9]
    # Calculamos la posición vertical de las 4 interfaces entre las 5 capas
    interfaces = np.cumsum(np.concatenate(([0], thicknesses))) - np.sum(thicknesses)/2
    
    # --- SECCIÓN DE CORTINA (CURTAIN SECTION) ---
    tvd_grid = np.linspace(-60, 60, 200)
    z_map = np.zeros((len(tvd_grid), len(df)))
    for i, z in enumerate(tvd_grid):
        idx = np.searchsorted(interfaces, z)
        z_map[i, :] = p[min(idx, 4)]

    fig_c = go.Figure()

    # 1. El Shading (Mapa de Calor)
    fig_c.add_trace(go.Heatmap(
        z=np.log10(z_map), x=df['MD'], y=tvd_grid,
        colorscale=GEO_COLORS, zsmooth='best', colorbar=dict(title="Log Res")
    ))

    # 2. Dibujar Líneas de Frontera (Interfaces)
    for i, inter in enumerate(interfaces):
        color = "white" if i in [1, 2] else "rgba(0,0,0,0.3)" # Resaltar Techo/Base del reservorio
        fig_c.add_trace(go.Scatter(
            x=df['MD'], y=np.full_like(df['MD'], inter),
            mode='lines', line=dict(color=color, width=1, dash='dash'),
            name=f"Interfaz {i+1}", showlegend=False
        ))

    # 3. Trayectoria y Etiquetas
    well_y = (df['MD'].values - df['MD'].values[0]) * np.tan(np.radians(user_inc - 90 - user_dip))
    fig_c.add_trace(go.Scatter(x=df['MD'], y=well_y, name="Trayectoria", line=dict(color='black', width=4)))
    
    # Etiquetas de texto para claridad
    fig_c.add_annotation(x=df['MD'].iloc[0], y=interfaces[1], text="TECHO (TOP)", font=dict(color="white"), showarrow=False)
    fig_c.add_annotation(x=df['MD'].iloc[0], y=interfaces[2], text="BASE", font=dict(color="white"), showarrow=False)

    fig_c.update_layout(height=650, title="Sección Estructural con Detalle de Capas (Reach 50ft)", 
                        yaxis_title="TVD Relativo (ft)", template="plotly_white")
    st.plotly_chart(fig_c, use_container_width=True)

    # Alerta de proximidad dinámica
    dist_top = abs(well_y[-1] - interfaces[1])
    if dist_top < 5:
        st.error(f"⚠️ ¡CUIDADO! La broca está a {dist_top:.1f} ft del TECHO.")
