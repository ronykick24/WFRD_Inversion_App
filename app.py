import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import clean_wfrd_data
from engine_wfrd import WFRD_Engine_Core

st.set_page_config(layout="wide", page_title="WFRD Geo-Pilot Pro")

# --- CONFIGURACIÓN DE COLORES Y ALGORITMO ---
st.sidebar.title("🎨 Visualización y Control")
color_theme = st.sidebar.selectbox("Paleta de Colores", ["Turbo", "Electric", "Viridis", "Plasma", "Cividis"])
algo_iters = st.sidebar.slider("Iteraciones (Precisión)", 20, 200, 50)
user_dip = st.sidebar.slider("DIP (Buzamiento) [(-) Asc / (+) Desc]", -10.0, 10.0, 0.0)

uploaded_file = st.file_uploader("Cargar Registro TSV", type=["tsv"])

if uploaded_file:
    raw_df = pd.read_csv(uploaded_file, sep='\t')
    raw_df.columns = [c.upper() for c in raw_df.columns]
    df = clean_wfrd_data(raw_df)
    
    engine = WFRD_Engine_Core()
    last_inc = df['INC'].iloc[-1]
    
    # Inversión de capas
    p, misfit = engine.solve("Estocástico", algo_iters, df['AD2_GW6'].values, df['MD'].values, last_inc, user_dip)
    
    # --- CÁLCULO DE ESPESORES Y DTBss ---
    # Interfaces: El reservorio principal es la Capa 4 (entre interf 2 y 3)
    thick = p[5:9]
    interfaces = np.cumsum(np.concatenate(([0], thick))) - np.sum(thick[:3])
    dtbss_top = interfaces[2]
    dtbss_base = interfaces[3]
    pay_thickness = thick[2] # Espesor de la capa 4

    # --- MÉTRICAS DE GEONAVEGACIÓN ---
    st.markdown(f"### 🚩 Estado de Perforación (NBI: {last_inc - user_dip:.1f}°)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("DTBss TOP (Techo)", f"{abs(dtbss_top):.1f} ft")
    m2.metric("DTBss BASE", f"{abs(dtbss_base):.1f} ft")
    m3.metric("Espesor Reservorio", f"{pay_thickness:.1f} ft")
    
    if abs(dtbss_top) < 4: m4.error("🚨 CRÍTICO: TECHO")
    elif abs(dtbss_base) < 4: m4.error("🚨 CRÍTICO: BASE")
    else: m4.success("💎 EN ZONA DE PAGO")

    # --- TRACKS VERTICALES ---
    fig_v = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=("Resistividad", "Geodirección Q"))
    for c in ['AD2_GW6', 'PD2_GW6']: 
        if c in df.columns: fig_v.add_trace(go.Scatter(x=df[c], y=df['MD'], name=c), row=1, col=1)
    for q in ['QPD2', 'QPD4', 'QPU1']: 
        if q in df.columns: fig_v.add_trace(go.Scatter(x=df[q], y=df['MD'], name=q), row=1, col=2)
    fig_v.update_yaxes(autorange="reversed")
    fig_v.update_xaxes(type="log", col=1)
    st.plotly_chart(fig_v, use_container_width=True)

    # --- CURTAIN SECTION: MODELO VS TRAYECTORIA ---
    tvd_grid = np.linspace(-60, 60, 150)
    # El modelo de fondo se inclina según el DIP
    dip_offset = -(df['MD'].values - df['MD'].values[0]) * np.tan(np.radians(user_dip))
    
    z_map = np.zeros((len(tvd_grid), len(df)))
    for j in range(len(df)):
        curr_ints = interfaces + dip_offset[j]
        idx = np.searchsorted(curr_ints, tvd_grid)
        z_map[:, j] = p[np.clip(idx, 0, 4)]

    fig_c = go.Figure()
    fig_c.add_trace(go.Heatmap(z=np.log10(z_map), x=df['MD'], y=tvd_grid, colorscale=color_theme))
    
    # Trayectoria real del pozo (reflejando la inclinación del registro)
    well_path = np.zeros(len(df)) # Aquí podrías integrar el cálculo de TVD real si lo deseas
    fig_c.add_trace(go.Scatter(x=df['MD'], y=well_path, name="Trayectoria", line=dict(color='white', width=4)))

    # Líneas de espesor y DTBss en el gráfico
    for i, inter in enumerate(interfaces):
        color = "rgba(255,255,255,0.5)" if i in [2,3] else "rgba(0,0,0,0.2)"
        fig_c.add_trace(go.Scatter(x=df['MD'], y=np.full(len(df), inter) + dip_offset, 
                                 mode='lines', line=dict(color=color, dash='dash'), showlegend=False))

    fig_c.update_layout(height=600, title="Sección Estructural Dinámica", yaxis_title="Relativo al Pozo (ft)", template="plotly_dark")
    st.plotly_chart(fig_c, use_container_width=True)
