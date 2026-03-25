import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils import clean_wfrd_data
from engine_wfrd import StochasticInversion

st.set_page_config(page_title="WFRD GuideWave Geosteering", layout="wide")

# Paleta Profesional: Azul (Conductivo) a Rojo (Resistivo)
COLOR_SCALE = [[0, '#00008B'], [0.2, '#4169E1'], [0.5, '#DCDCDC'], [0.8, '#FF4500'], [1, '#8B0000']]

st.title("🚜 WFRD GuideWave: Trayectoria y Cortina Multicapa")

uploaded_file = st.file_uploader("Cargar Registro (.tsv)", type=["tsv"])

if uploaded_file:
    df = clean_wfrd_data(pd.read_csv(uploaded_file, sep='\t'))
    
    # --- 1. CÁLCULO DE TRAYECTORIA (Simulación TVD) ---
    # Calculamos el TVD relativo para dibujar la línea del pozo
    df['Delta_MD'] = df['MD'].diff().fillna(0)
    df['TVD_Rel'] = (df['Delta_MD'] * np.cos(np.radians(df['INC']))).cumsum()

    # --- 2. GENERACIÓN DE CAPAS (33ft a 50ft) ---
    # Creamos un mallado para representar el espacio alrededor del pozo
    md_coords = df['MD'].values
    # Definimos 5 capas de investigación (offset radial desde el pozo)
    offsets = np.array([-50, -33, 0, 33, 50]) 
    
    # Mapeamos las curvas a estas profundidades
    # Usamos AD2 para 33ft, AU1 para 50ft. PD2 para capas cercanas.
    z_matrix = []
    for off in offsets:
        if off == 0: z_matrix.append(df['AD2_GW6']) # El pozo
        elif abs(off) <= 33: z_matrix.append(df['AD4_GW6']) # Capas medias
        else: z_matrix.append(df['AU1_GW6']) # Capas profundas (50ft)
    
    z_matrix = np.array(z_matrix)

    # --- 3. VISUALIZACIÓN DE CURTAIN SECTION CON TRAYECTORIA ---
    st.subheader("Simulación de Geonavegación: Trayectoria vs Formación")
    
    fig = go.Figure()

    # Añadir la Cortina de Capas (Heatmap)
    fig.add_trace(go.Heatmap(
        z=z_matrix,
        x=md_coords,
        y=offsets,
        colorscale=COLOR_SCALE,
        zsmooth='best',
        colorbar=dict(title="Resistividad", len=0.4)
    ))

    # Añadir la Línea de la Trayectoria del Pozo
    # La graficamos sobre el eje Y=0 para ver cómo "cruza" las capas
    fig.add_trace(go.Scatter(
        x=df['MD'],
        y=np.zeros(len(df)), # Línea en el centro de la herramienta
        mode='lines',
        line=dict(color='white', width=3, dash='dash'),
        name='Trayectoria del Pozo'
    ))

    fig.update_layout(
        xaxis_title="Profundidad Medida (MD) [ft]",
        yaxis_title="Distancia Radial al Pozo [ft] (Up/Down)",
        height=600,
        template="plotly_dark",
        yaxis=dict(range=[-60, 60]) # Ver el alcance total de 50ft
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- 4. PANEL DE CONTROL DE DATOS ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Inclinación Actual", f"{df['INC'].iloc[-1]}°")
    with col2:
        st.metric("Alcance Máximo", "50 ft")
    with col3:
        st.metric("Puntos Procesados", len(df))
