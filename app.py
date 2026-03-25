import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils import clean_wfrd_data
from engine_wfrd import WFRD_Simulator

st.set_page_config(layout="wide", page_title="WFRD 2D Inversion")

# Paleta Geológica Realista: Azul (Agua) -> Blanco (Roca) -> Amarillo/Rojo (Petróleo)
GEO_COLORS = [
    [0.0, 'rgb(0, 0, 139)'],    # Shale/Agua
    [0.2, 'rgb(173, 216, 230)'], # Transición
    [0.5, 'rgb(240, 240, 240)'], # Roca base
    [0.8, 'rgb(255, 165, 0)'],   # Oil Sand
    [1.0, 'rgb(139, 0, 0)']      # High Res Oil/Gas
]

st.title("🛡️ WFRD Geosteering Pro: Inversión Estocástica 2D")

uploaded_file = st.file_uploader("Subir Archivo TSV", type=["tsv"])

if uploaded_file:
    df = clean_wfrd_data(pd.read_csv(uploaded_file, sep='\t'))
    
    # Ejecutar motor de simulación
    sim = WFRD_Simulator()
    with st.spinner('Ejecutando Inversión Estocástica de 5 capas...'):
        p, misfit = sim.solve(df['AD2_GW6'].values, df['MD'].values, df['INC'].mean())
    
    # Extraer parámetros invertidos
    res_layers = p[:5]
    thicknesses = p[5:9]
    anisotropy = p[9]
    predicted_dip = p[10]

    # --- TRACKS SUPERIORES (MISFIT Y SENSIBILIDAD) ---
    c1, c2 = st.columns([1, 3])
    c1.metric("Misfit de Inversión", f"{misfit:.4f}")
    c1.metric("Buzamiento (Dip)", f"{predicted_dip:.2f}°")
    
    fig_t = go.Figure()
    fig_t.add_trace(go.Scatter(x=df['MD'], y=df['AD2_GW6'], name="Medido (33ft)", line=dict(color='black')))
    fig_t.add_trace(go.Scatter(x=df['MD'], y=sim.forward_model_2D(p, df['MD'].values, df['INC'].mean()), 
                               name="Simulado (Modelo)", line=dict(color='red', dash='dash')))
    fig_t.update_layout(height=250, title="Validación de Modelo vs Datos Reales", template="none")
    c2.plotly_chart(fig_t, use_container_width=True)

    # --- CURTAIN SECTION (SECCIÓN ESTRUCTURAL) ---
    st.subheader("Simulación 2D: Trayectoria cruzando 5 capas")
    
    # Crear malla vertical (TVD)
    tvd_grid = np.linspace(-60, 60, 120)
    interfaces = np.cumsum(np.concatenate(([0], thicknesses))) - np.sum(thicknesses)/2
    
    # Construir el mapa de formación inclinado por el DIP
    z_map = np.zeros((len(tvd_grid), len(df)))
    for i, z_val in enumerate(tvd_grid):
        # Encontrar en qué capa cae este punto Z
        l_idx = np.searchsorted(interfaces, z_val)
        z_map[i, :] = res_layers[min(l_idx, 4)]

    fig_c = go.Figure(data=go.Heatmap(
        z=np.log10(z_map), x=df['MD'], y=tvd_grid,
        colorscale=GEO_COLORS, zsmooth='best',
        colorbar=dict(title="Log10 Res")
    ))

    # Simular trayectoria a 85° cruzando las capas
    # El pozo se mueve en TVD relativo según (Inc - 90 - Dip)
    rel_angle = np.radians(df['INC'].mean() - 90 - predicted_dip)
    well_path = (df['MD'] - df['MD'].min()) * np.tan(rel_angle)
    
    fig_c.add_trace(go.Scatter(x=df['MD'], y=well_path, name="Trayectoria Pozo", 
                               line=dict(color='white', width=4, dash='solid')))

    fig_c.update_layout(height=600, yaxis_title="TVD Relativo (ft)", xaxis_title="MD (ft)")
    st.plotly_chart(fig_c, use_container_width=True)
