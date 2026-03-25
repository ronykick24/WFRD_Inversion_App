import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils import clean_wfrd_data
from engine_wfrd import StochasticInversion

st.set_page_config(layout="wide")

# Paleta específica solicitada
COLOR_SCALE = [[0, 'blue'], [0.4, 'lightblue'], [0.5, 'white'], [0.7, 'orange'], [1, 'red']]

st.title("🧩 Simulación Física WFRD: Espesor de Capas y Anisotropía")

uploaded_file = st.file_uploader("Cargar Datos (.tsv)", type=["tsv"])

if uploaded_file:
    df = clean_wfrd_data(pd.read_csv(uploaded_file, sep='\t'))
    
    # Supongamos una inclinación de 85° constante del archivo
    inc = df['INC'].mean() 
    
    # Ejecutar Inversión para hallar el ESPESOR de la capa
    engine = StochasticInversion()
    # Invertimos sobre un bloque de datos para mayor estabilidad
    params = engine.run_stochastic_inversion(df['AD2_GW6'].values, df['MD'].values, inc)
    
    rh1, rh2, thickness, contact_z, lan = params

    # --- CONSTRUCCIÓN DE LA SECCIÓN ESTRUCTURAL ---
    # Creamos un eje vertical de TVD (espesor real)
    tvd_grid = np.linspace(contact_z - 40, contact_z + thickness + 40, 100)
    md_grid = df['MD'].values
    
    # Definimos la geometría de las capas
    # Capa 1: Arriba del contacto (Shale)
    # Capa 2: Entre contacto y contacto + espesor (Pay Zone)
    # Capa 3: Debajo (Base)
    res_map = np.zeros((len(tvd_grid), len(md_grid)))
    
    for i, z in enumerate(tvd_grid):
        if z < contact_z:
            res_map[i, :] = rh1
        elif contact_z <= z <= (contact_z + thickness):
            res_map[i, :] = rh2
        else:
            res_map[i, :] = rh1 # Base conductiva

    # --- GRÁFICO DE GEONAVEGACIÓN REAL ---
    fig = go.Figure()

    # 1. Dibujar las capas con sus espesores calculados
    fig.add_trace(go.Heatmap(
        z=res_map, x=md_grid, y=tvd_grid,
        colorscale=COLOR_SCALE,
        colorbar=dict(title="Resistividad (Ohm-m)")
    ))

    # 2. Trayectoria del Pozo a 85° (Cruzando las capas)
    pozo_tvd = (df['MD'] - df['MD'].min()) * np.cos(np.radians(inc)) + (contact_z - 10)
    fig.add_trace(go.Scatter(
        x=df['MD'], y=pozo_tvd,
        mode='lines', line=dict(color='black', width=3),
        name="Trayectoria del Pozo (85°)"
    ))

    fig.update_layout(
        title=f"Inversión: Capa de {thickness:.1f} ft de espesor | Anisotropía λ={lan:.2f}",
        xaxis_title="Distancia Horizontal (MD)",
        yaxis_title="Profundidad Vertical (TVD)",
        yaxis=dict(autorange='reversed'), # Profundidad hacia abajo
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)
