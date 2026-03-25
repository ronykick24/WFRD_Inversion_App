import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from engine_wfrd import run_ahta_inversion
from physics_engine import calculate_tst

st.set_page_config(layout="wide", page_title="WFRD Master Dashboard")

# Gestión de Estado
if 'shift' not in st.session_state: st.session_state.shift = 0.0
if 'dip' not in st.session_state: st.session_state.dip = 0.0

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("well_logs.tsv", sep='\t')
        for c in ['MD', 'INC']: df[c] = pd.to_numeric(df[c], errors='coerce')
        return df.dropna(subset=['MD'])
    except: return pd.DataFrame()

df = load_data()

# Modelo de capas WFRD (AHTA Standard)
layers = [
    {"name": "Overburden", "tst": 20, "rh": 4, "rv": 8, "color": "#4b2c20"},
    {"name": "Upper Seal", "tst": 12, "rh": 2, "rv": 4, "color": "#757575"},
    {"name": "RESERVOIR", "tst": 35, "rh": 160, "rv": 240, "color": "#FFD700"},
    {"name": "Basal/OWC", "tst": 45, "rh": 15, "rv": 25, "color": "#00008B"}
]

# --- SIDEBAR: CONTROLES ---
with st.sidebar:
    st.header("🎮 Geosteering Control")
    st.session_state.dip = st.slider("DIP (°)", -15.0, 15.0, float(st.session_state.dip), 0.1)
    st.session_state.shift = st.slider("DTBss / Shift (ft)", -60.0, 60.0, float(st.session_state.shift), 0.5)
    
    if not df.empty and st.button("🚀 Run Inversion (1000 iter)"):
        res_col = [c for c in df.columns if 'RAD' in c or 'RES' in c][0]
        # Inversión basada en los últimos datos (Look-ahead)
        b_s, b_d = run_ahta_inversion(df[res_col].tail(40), df['INC'].tail(40), layers)
        st.session_state.shift, st.session_state.dip = b_s, b_d

# --- VISUALIZACIÓN PRINCIPAL ---
col_plot, col_bin = st.columns([0.7, 0.3])

with col_plot:
    st.subheader("Sección Estructural con Look-Ahead (200 ft)")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.3, 0.7])
    
    md = df['MD'].values
    z_mesh = np.linspace(-55, 55, 110)
    grid = np.zeros((len(z_mesh), len(md)))
    dip_rad = np.radians(st.session_state.dip)

    # Cálculo de la Cortina Estructural
    for i, x in enumerate(md):
        l_shift = st.session_state.shift + (x * np.tan(dip_rad))
        for j, z in enumerate(z_mesh):
            rel_z = z - l_shift
            cum = 0
            val = layers[-1]['rh']
            for ly in layers:
                if rel_z >= -cum - ly['tst'] and rel_z <= -cum:
                    val = ly['rh']
                    break
                cum += ly['tst']
            grid[j, i] = val

    # Renderizado de Heatmap y Wellbore
    fig.add_trace(go.Heatmap(x=md, y=z_mesh, z=grid, colorscale='Cividis', showscale=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=md, y=np.zeros_like(md), name="Pozo", line=dict(color='lime', width=3)), row=2, col=1)
    
    # Proyección Look-Ahead 200ft
    md_ahead = np.linspace(md[-1], md[-1]+200, 15)
    fig.add_trace(go.Scatter(x=md_ahead, y=np.zeros_like(md_ahead), name="Look-Ahead", line=dict(color='lime', dash='dash')), row=2, col=1)

    # Track de Log de Resistividad
    res_c = [c for c in df.columns if 'RAD' in c or 'RES' in c][0]
    fig.add_trace(go.Scatter(x=md, y=df[res_c], name="Log Real", line=dict(color='white')), row=1, col=1)
    
    fig.update_layout(height=800, template="plotly_dark", showlegend=False)
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(autorange="reversed", title="Vertical Offset (ft)", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

with col_bin:
    st.subheader("WFRD Bin Plot (16 Sectors)")
    # Respuesta Azimutal Proactiva
    theta = np.linspace(0, 360, 16, endpoint=False)
    intensidad = [np.exp(-abs(st.session_state.shift)/15) * np.abs(np.cos(np.radians(t))) for t in theta]
    
    fig_polar = go.Figure(go.Barpolar(r=intensidad, theta=theta, marker_color=intensidad, colorscale='Viridis'))
    fig_polar.update_layout(template="plotly_dark", height=400, polar=dict(radialaxis=dict(visible=False)))
    st.plotly_chart(fig_polar, use_container_width=True)
    
    # Datos de Navegación
    tst_actual = calculate_tst(layers[2]['tst'], st.session_state.dip)
    st.info(f"**DTBss:** {st.session_state.shift:.2f} ft")
    st.info(f"**DIP:** {st.session_state.dip:.2f} °")
    st.success(f"**TST Reservorio:** {tst_actual:.1f} ft")
