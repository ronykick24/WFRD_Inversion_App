import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from engine_wfrd import run_proactive_inversion 
from physics_engine import calculate_3d_horns, get_geo_metrics
from utils import get_wfrd_palette

st.set_page_config(layout="wide", page_title="WFRD Master Suite")

if 'shift' not in st.session_state: st.session_state.shift = 0.0
if 'dip' not in st.session_state: st.session_state.dip = 0.0

@st.cache_data
def load_data():
    try:
        data = pd.read_csv("well_logs.tsv", sep='\t')
        for col in ['MD', 'INC']: data[col] = pd.to_numeric(data[col], errors='coerce')
        return data.dropna(subset=['MD'])
    except: return pd.DataFrame()

df = load_data()

layers = [
    {"name": "Overburden", "tst": 20, "rh": 20, "rv": 30, "color": "#4b2c20"},
    {"name": "Upper Seal", "tst": 8, "rh": 2, "rv": 4, "color": "#757575"},
    {"name": "TARGET RESERVOIR", "tst": 30, "rh": 120, "rv": 180, "color": "#FFD700"},
    {"name": "Basal / OWC", "tst": 50, "rh": 40, "rv": 60, "color": "#00008B"}
]

with st.sidebar:
    st.header("🔍 WFRD Control Center")
    st.session_state.dip = st.slider("DIP (°)", -15.0, 15.0, st.session_state.dip)
    st.session_state.shift = st.slider("Shift (ft)", -50.0, 50.0, st.session_state.shift)
    
    if not df.empty and st.button("🚀 Inversión Estocástica"):
        prog = st.progress(0, text="Iniciando...")
        res_col = [c for c in df.columns if 'RAD' in c or 'RES' in c][0]
        b_s, b_d = run_proactive_inversion(df[res_col].tail(20).values, df['INC'].tail(20).values, layers, prog)
        st.session_state.shift, st.session_state.dip = float(b_s), float(b_d)

st.title("Weatherford GuideWave Master Dashboard")

if not df.empty:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.3, 0.7])
    
    # --- TRACK 1: LOGS (Real vs 3D vs 2D) ---
    res_col = [c for c in df.columns if 'RAD' in c or 'RES' in c][0]
    fig.add_trace(go.Scatter(x=df['MD'], y=df[res_col], name="Measured (Real)", line=dict(color='white')), row=1, col=1)
    
    synth_3d = [calculate_3d_horns(layers[2]['rh'], layers[2]['rv'], inc, st.session_state.dip, st.session_state.shift) for inc in df['INC']]
    fig.add_trace(go.Scatter(x=df['MD'], y=synth_3d, name="Synth 3D (Aniso)", line=dict(color='red', dash='dot')), row=1, col=1)
    
    # --- TRACK 2: CORTINA + LOOK AHEAD ---
    md_array = df['MD'].values
    y_base = -5160.0 + st.session_state.shift
    
    # Dibujar Capas con Paleta OWC
    for i, ly in enumerate(layers):
        y_layer = y_base + (md_array * np.tan(np.radians(st.session_state.dip)))
        fig.add_trace(go.Scatter(x=md_array, y=y_layer, fill='toself', fillcolor=ly['color'], line_width=0, name=ly['name']), row=2, col=1)
        
        # LOOK AHEAD 200FT: Proyección de capas hacia adelante
        md_ahead = np.linspace(md_array[-1], md_array[-1] + 200, 10)
        y_ahead = y_layer[-1] + ((md_ahead - md_array[-1]) * np.tan(np.radians(st.session_state.dip)))
        fig.add_trace(go.Scatter(x=md_ahead, y=y_ahead, line=dict(color=ly['color'], dash='dash'), showlegend=False), row=2, col=1)
        
        y_base -= ly['tst']

    # Trayectoria del Pozo (Referencia)
    fig.add_trace(go.Scatter(x=md_array, y=np.full_like(md_array, -5160), name="Wellbore", line=dict(color='lime', width=2)), row=2, col=1)

    fig.update_layout(height=900, template="plotly_dark", showlegend=True, coloraxis={'colorscale': get_wfrd_palette()})
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # MÉTRICAS: DTBss como punto estructural
    tvdss, dtbss_point = get_geo_metrics(md_array[-1], df['INC'].iloc[-1], st.session_state.dip, st.session_state.shift, 30)
    c1, c2, c3 = st.columns(3)
    c1.metric("DTBss (Límite Estructural)", f"{dtbss_point:.1f} ft")
    c2.metric("Buzamiento Actual", f"{st.session_state.dip}°")
    c3.metric("TVDss @ Bit", f"{tvdss:.1f} ft")
