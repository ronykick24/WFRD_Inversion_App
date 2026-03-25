import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from engine_wfrd import run_ahta_inversion
from physics_engine import get_geo_metrics
from utils import get_wfrd_palette

st.set_page_config(layout="wide", page_title="Weatherford GuideWave AHTA")

if 'shift' not in st.session_state: st.session_state.shift = 0.0
if 'dip' not in st.session_state: st.session_state.dip = 0.0

@st.cache_data
def load_data():
    try:
        data = pd.read_csv("well_logs.tsv", sep='\t')
        for c in ['MD', 'INC']: data[c] = pd.to_numeric(data[c], errors='coerce')
        return data.dropna(subset=['MD'])
    except: return pd.DataFrame()

df = load_data()

# Modelo de Capas (Basado en el manual y tus imágenes)
layers = [
    {"name": "Sello Superior", "tst": 15, "rh": 4, "rv": 8, "color": "#4b2c20"},
    {"name": "Transición", "tst": 10, "rh": 25, "rv": 40, "color": "#757575"},
    {"name": "TARGET SAND", "tst": 35, "rh": 180, "rv": 280, "color": "#FFD700"},
    {"name": "Basal (OWC)", "tst": 45, "rh": 15, "rv": 25, "color": "#00008B"}
]

# --- UI CONTROLS ---
with st.sidebar:
    st.image("https://www.weatherford.com/logos/logo.png", width=200) # Opcional
    st.header("AHTA Control Panel")
    st.session_state.dip = st.slider("Buzamiento / DIP (°)", -15.0, 15.0, float(st.session_state.dip), 0.1)
    st.session_state.shift = st.slider("DTBss / Shift (ft)", -50.0, 50.0, float(st.session_state.shift), 0.5)
    
    if st.button("🚀 Inversión Estocástica AHTA"):
        prog = st.progress(0)
        res_col = [c for c in df.columns if 'RAD' in c or 'RES' in c][0]
        b_s, b_d = run_ahta_inversion(df[res_col].tail(30).values, df['INC'].tail(30).values, layers, prog)
        st.session_state.shift, st.session_state.dip = float(b_s), float(b_d)

# --- VISUALIZACIÓN DE SECCIÓN ESTRUCTURAL ---
st.title("WFRD GuideWave Azimuthal Resistivity")

if not df.empty:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.3, 0.7])
    
    md_x = df['MD'].values
    z_y = np.linspace(-45, 45, 90) # Rango de investigación de la cortina
    
    # Generar la Cortina Estructural Inclinada (Heatmap)
    grid = np.zeros((len(z_y), len(md_x)))
    dip_rad = np.radians(st.session_state.dip)
    
    for i, x in enumerate(md_x):
        # Desplazamiento estructural (deformación de la capa)
        structural_offset = st.session_state.shift + (x * np.tan(dip_rad))
        for j, z_val in enumerate(z_y):
            rel_pos = z_val - structural_offset
            cum = 0
            val = layers[-1]['rh']
            for ly in layers:
                if rel_pos >= -cum - ly['tst'] and rel_pos <= -cum:
                    val = ly['rh']
                    break
                cum += ly['tst']
            grid[j, i] = val

    # Renderizado Cortina
    fig.add_trace(go.Heatmap(x=md_x, y=z_y, z=grid, colorscale=get_wfrd_palette(), showscale=False), row=2, col=1)
    
    # Wellbore y Proyección LOOK AHEAD 200FT
    fig.add_trace(go.Scatter(x=md_x, y=np.zeros_like(md_x), name="Wellbore", line=dict(color='lime', width=3)), row=2, col=1)
    md_ahead = np.linspace(md_x[-1], md_x[-1] + 200, 15)
    fig.add_trace(go.Scatter(x=md_ahead, y=np.zeros_like(md_ahead), name="Look-Ahead", line=dict(color='lime', dash='dash')), row=2, col=1)

    # Track de Logs (Física AHTA)
    res_col = [c for c in df.columns if 'RAD' in c or 'RES' in c][0]
    fig.add_trace(go.Scatter(x=md_x, y=df[res_col], name="Log Real", line=dict(color='white')), row=1, col=1)

    fig.update_layout(height=850, template="plotly_dark")
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(autorange="reversed", title="Structural Vertical Offset", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # MÉTRICAS FINALES
    tvdss, dtbss = get_geo_metrics(md_x[-1], df['INC'].iloc[-1], st.session_state.dip, st.session_state.shift)
    c1, c2, c3 = st.columns(3)
    c1.metric("DTBss (Boundary)", f"{dtbss:.1f} ft")
    c2.metric("DIP Estructural", f"{st.session_state.dip:.1f}°")
    c3.metric("TVDss @ Bit", f"{tvdss:.1f} ft")
