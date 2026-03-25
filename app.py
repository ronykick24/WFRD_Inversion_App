import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import clean_wfrd_data
from engine_wfrd import WFRD_Engine_Core

st.set_page_config(layout="wide", page_title="WFRD Geo-Master v3")

# PALETA SOLICITADA: Azul -> Amarillo -> Naranja -> Rojo (Degradé)
CUSTOM_GEO = [
    [0.0, "#00008B"],   # Azul Intenso (Arcilla/Sello)
    [0.3, "#00FFFF"],   # Cian (Transición)
    [0.6, "#FFFF00"],   # Amarillo (Arena)
    [0.8, "#FF8C00"],   # Naranja (Reservorio)
    [1.0, "#FF0000"]    # Rojo (Pay Zone)
]

st.sidebar.title("🎮 Panel de Control")
n_layers = st.sidebar.slider("Cantidad de Capas", 3, 7, 5)
algo_iters = st.sidebar.number_input("Iteraciones", 10, 500, 50)
user_dip = st.sidebar.slider("DIP [(-) Asc / (+) Desc]", -15.0, 15.0, 0.0)

uploaded_file = st.file_uploader("Cargar Registro TSV", type=["tsv"])

if uploaded_file:
    raw_df = pd.read_csv(uploaded_file, sep='\t')
    # Normalizar para encontrar QPD2, QPU1, etc.
    raw_df.columns = [c.upper() for c in raw_df.columns]
    df = clean_wfrd_data(raw_df)
    
    engine = WFRD_Engine_Core()
    last_inc = df['INC'].iloc[-1]
    
    # Inversión
    p, misfit = engine.solve(algo_iters, df['AD2_GW6'].values, df['MD'].values, last_inc, user_dip, n_layers)
    
    # Extraer parámetros
    res_vals = p[:n_layers]
    thick_vals = p[n_layers:2*n_layers-1]
    interfaces = np.cumsum(np.concatenate(([0], thick_vals))) - np.sum(thick_vals)/2
    
    # DTBss al Reservorio (Capa Central)
    target_idx = n_layers // 2
    dtbss_top = interfaces[target_idx]
    dtbss_base = interfaces[target_idx + 1]

    # --- LABELS Y ALERTAS ---
    st.markdown("### 🛰️ Monitoreo de Geonavegación")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("DTBss TOP", f"{abs(dtbss_top):.1f} ft", delta="TECHO")
    c2.metric("DTBss BASE", f"{abs(dtbss_base):.1f} ft", delta="BASE")
    c3.metric("ESPESOR CAPA", f"{thick_vals[target_idx-1]:.1f} ft")
    
    if abs(dtbss_top) < 5 or abs(dtbss_base) < 5:
        c4.error("🚨 ALERTA: PROXIMIDAD")
    else:
        c4.success("💎 DENTRO DE ARENA")

    # --- TRACKS VERTICALES (MOSTRANDO Q Y LABELS) ---
    fig_v = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=("Resistividades", "Geodirección (Q)"))
    
    # Track 1: Resistividades
    for c in ['AD2_GW6', 'PD2_GW6', 'AU1_GW6']:
        if c in df.columns: fig_v.add_trace(go.Scatter(x=df[c], y=df['MD'], name=c), row=1, col=1)
    
    # Track 2: Curvas Q (Asegurando que se vean QPD2, QPD4, QPU1)
    q_to_show = ['QPD2', 'QPD4', 'QPU1', 'QPU4']
    for q in q_to_show:
        if q in df.columns:
            fig_v.add_trace(go.Scatter(x=df[q], y=df['MD'], name=q, line=dict(dash='dot')), row=1, col=2)
    
    fig_v.update_yaxes(autorange="reversed")
    fig_v.update_xaxes(type="log", col=1)
    fig_v.update_layout(height=400, template="plotly_dark", showlegend=True)
    st.plotly_chart(fig_v, use_container_width=True)

    # --- CURTAIN SECTION CON PALETA PERSONALIZADA ---
    tvd_grid = np.linspace(-60, 60, 150)
    dip_offset = -(df['MD'].values - df['MD'].values[0]) * np.tan(np.radians(user_dip))
    
    z_map = np.zeros((len(tvd_grid), len(df)))
    for j in range(len(df)):
        curr_ints = interfaces + dip_offset[j]
        idx = np.searchsorted(curr_ints, tvd_grid)
        z_map[:, j] = res_vals[np.clip(idx, 0, n_layers-1)]

    fig_c = go.Figure()
    fig_c.add_trace(go.Heatmap(z=np.log10(z_map), x=df['MD'], y=tvd_grid, colorscale=CUSTOM_GEO, zsmooth='best'))
    fig_c.add_trace(go.Scatter(x=df['MD'], y=np.zeros(len(df)), name="Pozo", line=dict(color='white', width=4)))

    # Etiquetas de profundidad en el gráfico
    fig_c.add_annotation(x=df['MD'].iloc[-1], y=dtbss_top, text="TOP", showarrow=False, font=dict(color="white"))
    fig_c.add_annotation(x=df['MD'].iloc[-1], y=dtbss_base, text="BASE", showarrow=False, font=dict(color="white"))

    fig_c.update_layout(height=600, title=f"Sección Estructural ({n_layers} Capas)", yaxis_title="TVD Rel. (ft)", template="plotly_dark")
    st.plotly_chart(fig_c, use_container_width=True)
