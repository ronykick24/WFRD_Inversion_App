import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from engine_wfrd import WFRD_Simulator
from utils import clean_wfrd_data

# Paleta Geológica Real
RESERVOIR_COLORS = [
    [0.0, 'rgb(0, 0, 139)'],    # Agua / Arcilla (Conductivo)
    [0.15, 'rgb(173, 216, 230)'],# Transición Agua-Aceite
    [0.3, 'rgb(255, 255, 255)'], # Roca Limpia
    [0.6, 'rgb(255, 215, 0)'],   # Petróleo (Arena)
    [1.0, 'rgb(139, 0, 0)']      # Petróleo muy resistivo / Gas
]

st.set_page_config(layout="wide")
st.title("🛡️ WFRD Advanced Geosteering: 1D/2D Stochastic Inversion")

uploaded_file = st.file_uploader("Cargar Registro TSV", type=["tsv"])

if uploaded_file:
    df = clean_wfrd_data(pd.read_csv(uploaded_file, sep='\t'))
    sim = WFRD_Simulator()
    
    # Inversión Estocástica
    p, predicted_dip = sim.solve(df['AD2_GW6'].values, df['MD'].values, df['INC'].mean())[:-1], sim.solve(df['AD2_GW6'].values, df['MD'].values, df['INC'].mean())[-1]

    # --- SIMULACIÓN DE CAPAS (Sección Estructural) ---
    tvd_grid = np.linspace(-60, 60, 150)
    z_map = np.zeros((len(tvd_grid), len(df)))
    
    # Construcción de las 5 capas con sus espesores reales invertidos
    thicknesses = p[5:9]
    interfaces = np.cumsum(thicknesses) - np.sum(thicknesses)/2
    
    for i, z_val in enumerate(tvd_grid):
        layer_idx = np.searchsorted(interfaces, z_val)
        z_map[i, :] = p[min(layer_idx, 4)]

    # --- GRÁFICO PROFESIONAL ---
    fig = go.Figure()
    
    # Capas Geológicas
    fig.add_trace(go.Heatmap(
        z=np.log10(z_map), # Escala logarítmica para mejor shading
        x=df['MD'], y=tvd_grid,
        colorscale=RESERVOIR_COLORS,
        zsmooth='best',
        colorbar=dict(title="Log10 Resistividad")
    ))

    # Trayectoria del Pozo (Cruza las capas a 85°)
    # Calculamos el TVD acumulado real basado en la Inclinación y el Dip
    path_tvd = (df['MD'] - df['MD'].min()) * np.tan(np.radians(df['INC'].mean() - 90 - predicted_dip))
    
    fig.add_trace(go.Scatter(
        x=df['MD'], y=path_tvd,
        mode='lines+markers',
        line=dict(color='black', width=4),
        marker=dict(size=4, color='white'),
        name="Trayectoria (85° Inc)"
    ))

    fig.update_layout(
        title=f"Simulación de Cruce de Capas | Dip Predicho: {predicted_dip:.2f}° | Anisotropía: {p[9]:.2f}",
        xaxis_title="MD (ft)", yaxis_title="Distancia Vertical al Pozo (ft)",
        height=700, template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.info(f"El pozo está atravesando una capa de {p[layer_idx]:.2f} Ohm-m. Sensibilidad a 50ft activa.")
