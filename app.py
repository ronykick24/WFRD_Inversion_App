import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import clean_wfrd_data
from engine_wfrd import WFRD_Engine_Core

st.set_page_config(layout="wide", page_title="Geo-Master Pro V4")

# Paleta: Azul Intenso -> Amarillo -> Naranja -> Rojo
CUSTOM_GEO = [
    [0.0, "#00008B"], [0.2, "#0000FF"], [0.5, "#FFFF00"], 
    [0.8, "#FF8C00"], [1.0, "#FF0000"]
]

# --- SIDEBAR: BOTONES Y OPCIONES ---
st.sidebar.title("🕹️ Panel de Control")
calc_mode = st.sidebar.selectbox("Método de Inversión", 
    ["Estocástico Global", "Estocástico Local", "Determinístico"])
n_layers = st.sidebar.slider("Número de Capas", 3, 7, 5)
iters = st.sidebar.number_input("Iteraciones", 10, 1000, 100)
user_dip = st.sidebar.slider("DIP (Buzamiento)", -15.0, 15.0, 0.0)

uploaded_file = st.file_uploader("Cargar Registro TSV", type=["tsv"])

if uploaded_file:
    raw_df = pd.read_csv(uploaded_file, sep='\t')
    raw_df.columns = [c.upper() for c in raw_df.columns]
    df = clean_wfrd_data(raw_df)
    
    engine = WFRD_Engine_Core()
    last_inc = df['INC'].iloc[-1]
    
    # Ejecución de Inversión
    p, misfit = engine.solve(calc_mode, iters, df['AD2_GW6'].values, df['MD'].values, last_inc, user_dip, n_layers)
    
    # Desglose de Capas y DTBss
    res_vals = p[:n_layers]
    thick_vals = p[n_layers:2*n_layers-1]
    interfaces = np.cumsum(np.concatenate(([0], thick_vals))) - np.sum(thick_vals)/2
    
    target_idx = n_layers // 2
    dtbss_up = interfaces[target_idx]
    dtbss_down = interfaces[target_idx + 1]

    # --- DASHBOARD DE METRICAS ---
    st.subheader(f"📊 Análisis de Proximidad (NBI: {last_inc - user_dip:.1f}°)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("DTBss ARRIBA", f"{abs(dtbss_up):.1f} ft", delta="TECHO", delta_color="inverse")
    m2.metric("DTBss ABAJO", f"{abs(dtbss_down):.1f} ft", delta="BASE")
    m3.metric("ESPESOR NETO", f"{thick_vals[target_idx-1]:.1f} ft")
    
    if abs(dtbss_up) < 5 or abs(dtbss_down) < 5:
        m4.error("🚨 ALERTA: FUERA DE VENTANA")
    else:
        m4.success("💎 TARGET OPTIMIZADO")

    # --- TRACKS VERTICALES CON CURVAS Q ---
    fig_v = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=("Resistividad", "Geodirección (Q)"))
    res_list = ['AD2_GW6', 'PD2_GW6', 'AU1_GW6', 'PU1_GW6']
    q_list = ['QPD2', 'QPD4', 'QPU1', 'QPU4']
    
    for c in res_list:
        if c in df.columns: fig_v.add_trace(go.Scatter(x=df[c], y=df['MD'], name=c), row=1, col=1)
    for q in q_list:
        if q in df.columns: fig_v.add_trace(go.Scatter(x=df[q], y=df['MD'], name=q, line=dict(dash='dot')), row=1, col=2)
        
    fig_v.update_yaxes(autorange="reversed")
    fig_v.update_xaxes(type="log", col=1)
    fig_v.update_layout(height=400, template="plotly_dark")
    st.plotly_chart(fig_v, use_container_width=True)

    # --- CURTAIN SECTION CON DESGLOSE DE DTBss ---
    st.subheader("🌐 Sección de Cortina Estratigráfica")
    tvd_grid = np.linspace(-60, 60, 150)
    dip_offset = -(df['MD'].values - df['MD'].values[0]) * np.tan(np.radians(user_dip))
    
    z_map = np.zeros((len(tvd_grid), len(df)))
    for j in range(len(df)):
        curr_ints = interfaces + dip_offset[j]
        idx = np.searchsorted(curr_ints, tvd_grid)
        z_map[:, j] = res_vals[np.clip(idx, 0, n_layers-1)]

    fig_c = go.Figure()
    # Heatmap con paleta azul-amarillo-naranja-rojo
    fig_c.add_trace(go.Heatmap(z=np.log10(z_map), x=df['MD'], y=tvd_grid, colorscale=CUSTOM_GEO, zsmooth='best'))
    
    # Trayectoria del pozo
    fig_c.add_trace(go.Scatter(x=df['MD'], y=np.zeros(len(df)), name="Pozo", line=dict(color='white', width=4)))

    # DESGLOSE VISUAL DE DTBss (Etiquetas dinámicas)
    # Línea de Techo
    fig_c.add_trace(go.Scatter(x=df['MD'], y=np.full(len(df), dtbss_up) + dip_offset, 
                             name="Techo", line=dict(color='rgba(255,255,255,0.6)', dash='dash')))
    # Línea de Base
    fig_c.add_trace(go.Scatter(x=df['MD'], y=np.full(len(df), dtbss_down) + dip_offset, 
                             name="Base", line=dict(color='rgba(255,255,255,0.6)', dash='dash')))

    # Anotaciones de DTBss en la broca
    fig_c.add_annotation(x=df['MD'].iloc[-1], y=dtbss_up, text=f"↑ {abs(dtbss_up):.1f} ft", showarrow=True, arrowhead=1, font=dict(color="cyan"))
    fig_c.add_annotation(x=df['MD'].iloc[-1], y=dtbss_down, text=f"↓ {abs(dtbss_down):.1f} ft", showarrow=True, arrowhead=1, font=dict(color="yellow"))

    fig_c.update_layout(height=600, yaxis_title="TVD Rel. (ft)", template="plotly_dark")
    st.plotly_chart(fig_c, use_container_width=True)
