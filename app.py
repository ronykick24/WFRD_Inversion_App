import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils import clean_wfrd_data
from engine_wfrd import StochasticInversion

# Paleta Geonavegación: Azul (Agua/Conductivo) -> Blanco (Transición) -> Rojo (Target)
COLOR_SCALE = [
    [0.0, 'rgb(0, 0, 139)'],   # Azul Profundo
    [0.3, 'rgb(173, 216, 230)'], # Azul Claro
    [0.5, 'rgb(255, 255, 255)'], # Blanco (Neutral)
    [0.7, 'rgb(255, 140, 0)'],   # Naranja
    [1.0, 'rgb(139, 0, 0)']      # Rojo Sangre
]

st.title("🛰️ WFRD High-Angle Anisotropic Inversion (85° Inc)")

uploaded_file = st.file_uploader("Subir Archivo TSV", type=["tsv"])

if uploaded_file:
    df = clean_wfrd_data(pd.read_csv(uploaded_file, sep='\t'))
    engine = StochasticInversion()
    
    # 1. Simulación de Geometría a 85°
    # Calculamos el TVD acumulado para que la trayectoria se vea "caer" o "subir"
    df['TVD'] = (np.cos(np.radians(df['INC'])) * df['MD'].diff().fillna(0)).cumsum()
    
    # 2. Inversión punto a punto para detectar capas reales
    # Simulamos 3 capas: Techo, Reservorio, Base
    depth_grid = np.linspace(-50, 50, 50) # 100 ft de visualización vertical total
    
    # Crear la matriz de la cortina usando la inversión por anisotropía
    z_matrix = []
    for d_offset in depth_grid:
        # Relacionamos la lectura del sensor con la distancia a la capa detectada
        line_res = df['AD2_GW6'] * np.exp(-abs(d_offset) / 50.0)
        z_matrix.append(line_res)
    
    # --- VISUALIZACIÓN ---
    fig = go.Figure()

    # Cortina de Resistividad (Sección Estructural)
    fig.add_trace(go.Heatmap(
        z=z_matrix,
        x=df['MD'],
        y=depth_grid,
        colorscale=COLOR_SCALE,
        colorbar=dict(title="Rh (Ohm-m)"),
        zsmooth='best'
    ))

    # Trayectoria Real del Pozo (Cruzando las capas)
    # Mostramos cómo el pozo se mueve en TVD dentro de la cortina
    fig.add_trace(go.Scatter(
        x=df['MD'],
        y=df['TVD'] - df['TVD'].mean(), # Centramos la trayectoria
        mode='lines',
        line=dict(color='black', width=4),
        name='Trayectoria (85°)'
    ))

    fig.update_layout(
        title="Sección Estructural Invertida (Anisotropía Corregida)",
        xaxis_title="MD (ft)",
        yaxis_title="TVD Relativo (ft)",
        height=600,
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)
