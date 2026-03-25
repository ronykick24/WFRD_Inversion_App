import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils import clean_wfrd_data
from engine_wfrd import StochasticInversion

st.set_page_config(page_title="WFRD GuideWave Pro Panel", layout="wide")

# Paleta de colores solicitada: Azul (Conductivo) -> Rojo (Resistivo)
COLOR_SCALE = [
    [0.0, '#00008B'],   # Azul muy oscuro (Arcilla/Agua)
    [0.2, '#4169E1'],   # Azul real
    [0.5, '#DCDCDC'],   # Gris (Transición)
    [0.8, '#FF4500'],   # Naranja-Rojo
    [1.0, '#8B0000']    # Rojo oscuro (Hidrocarburo)
]

st.title("🛡️ WFRD GuideWave: Multi-Layer Curtain Section")

uploaded_file = st.file_uploader("Cargar Registro WFRD (.tsv)", type=["tsv"])

if uploaded_file:
    # 1. Procesamiento de datos
    df_raw = pd.read_csv(uploaded_file, sep='\t')
    df = clean_wfrd_data(df_raw)
    
    # 2. Selección de curvas para la inversión
    st.sidebar.header("Configuración de Capas")
    selected_curves = st.sidebar.multiselect(
        "Curvas para Inversión Estocástica",
        ['AD2_GW6', 'AD4_GW6', 'AU1_GW6', 'PD2_GW6', 'PD4_GW6', 'PU1_GW6'],
        default=['AD2_GW6', 'AD4_GW6', 'AU1_GW6']
    )

    # 3. Creación de la Curtain Section (Mapa de Calor Multicapa)
    st.subheader("Sección de Cortina (Curtain Section) - Alcance 50ft")
    
    # Preparamos los datos para la cortina (transponemos sensores como capas de profundidad)
    curtain_data = df[selected_curves].values.T
    
    fig = go.Figure(data=go.Heatmap(
        z=curtain_data,
        x=df['MD'],
        y=['33ft', '40ft', '50ft'][:len(selected_curves)], # Mapeo físico de WFRD
        colorscale=COLOR_SCALE,
        zsmooth='best',
        colorbar=dict(title="Resistividad / Señal")
    ))

    fig.update_layout(
        xaxis_title="Profundidad Medida (MD) [ft]",
        yaxis_title="Profundidad de Investigación (Radial)",
        height=500,
        template="plotly_dark"
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # 4. Comparativa de Curvas (Logs Convencionales)
    st.subheader("Registros Azimutales y de Fase")
    fig_logs = go.Figure()
    for col in selected_curves:
        fig_logs.add_trace(go.Scatter(x=df['MD'], y=df[col], name=col))
    
    fig_logs.update_layout(height=400, template="plotly_dark", xaxis_title="MD")
    st.plotly_chart(fig_logs, use_container_width=True)

    # 5. Detección Automática de Capas (D1 - Distance to Boundary)
    if 'D1' in df.columns:
        st.info(f"Análisis de Geonavegación: Capa detectada en promedio a {df['D1'].mean():.2f} ft del sensor.")
