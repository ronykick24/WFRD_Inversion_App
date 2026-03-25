import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from engine_wfrd import run_proactive_inversion 
from physics_engine import calculate_3d_horns, get_geo_metrics
from utils import get_wfrd_palette

st.set_page_config(layout="wide", page_title="WFRD Master Suite")

# --- PERSISTENCIA ---
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

# Modelo de Capas (WFRD Standar)
layers = [
    {"name": "Overburden", "tst": 20, "rh": 20, "rv": 30, "color": "#4b2c20"},
    {"name": "Upper Seal", "tst": 10, "rh": 2, "rv": 5, "color": "#757575"},
    {"name": "TARGET RES", "tst": 30, "rh": 150, "rv": 210, "color": "#FFD700"},
    {"name": "Basal / OWC", "tst": 50, "rh": 40, "rv": 65, "color": "#00008B"}
]

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Geosteering Control")
    st.session_state.dip = st.slider("DIP Angle (°)", -15.0, 15.0, float(st.session_state.dip), 0.1)
    st.session_state.shift = st.slider("Shift / DTBss (ft)", -60.0, 60.0, float(st.session_state.shift), 0.5)
    
    if not df.empty and st.button("🚀 Run Inversion (1000 iter)"):
        res_col = [c for c in df.columns if 'RAD' in c or 'RES' in c][0]
        # Inversión Estocástica Proactiva
        b_s, b_d = run_proactive_inversion(df[res_col].tail(30).values, df['INC'].tail(30).values, layers)
        st.session_state.shift, st.session_state.dip = float(b_s), float(b_d)

# --- VISUALIZACIÓN ---
if not df.empty:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.3, 0.7])
    
    md_x = df['MD'].values
    res_col = [c for c in df.columns if 'RAD' in c or 'RES' in c][0]
    
    # 1. Logs: Real vs Synth 3D
    fig.add_trace(go.Scatter(x=md_x, y=df[res_col], name="Log Real", line=dict(color='white')), row=1, col=1)
    
    # 2. CORTINA ESTRUCTURAL (Heatmap 2D)
    z_y = np.linspace(-50, 50, 100)
    grid = np.zeros((len(z_y), len(md_x)))
    dip_rad = np.radians(st.session_state.dip)
    
    for i, x in enumerate(md_x):
        local_shift = st.session_state.shift + (x * np.tan(dip_rad))
        for j, z in enumerate(z_y):
            rel_z = z - local_shift
            # Asignación multicapa
            cum = 0
            val = layers[-1]['rh']
            for ly in layers:
                if rel_z >= -cum - ly['tst'] and rel_z <= -cum:
                    val = ly['rh']
                    break
                cum += ly['tst']
            grid[j, i] = val

    fig.add_trace(go.Heatmap(x=md_x, y=z_y, z=grid, colorscale=get_wfrd_palette(), showscale=False), row=2, col=1)
    
    # Look Ahead 200ft
    md_ahead = np.linspace(md_x[-1], md_x[-1] + 200, 10)
    fig.add_trace(go.Scatter(x=md_ahead, y=np.zeros_like(md_ahead), line=dict(color='lime', dash='dash'), name="Look Ahead"), row=2, col=1)

    fig.update_layout(height=850, template="plotly_dark")
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # MÉTRICAS
    tvdss, dtbss = get_geo_metrics(md_x[-1], df['INC'].iloc[-1], st.session_state.dip, st.session_state.shift)
    st.metric("DTBss Estructural (Top)", f"{dtbss:.1f} ft")
