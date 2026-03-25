import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from engine_wfrd import run_proactive_inversion 
from physics_engine import calculate_3d_horns, get_geo_metrics
from utils import get_palettes

st.set_page_config(layout="wide", page_title="WFRD Master Suite")

# --- ESTADO DE SESIÓN ---
if 'shift' not in st.session_state: st.session_state.shift = 0.0
if 'dip' not in st.session_state: st.session_state.dip = 0.0
if 'history' not in st.session_state: st.session_state.history = []

@st.cache_data
def load_data():
    data = pd.read_csv("well_logs.tsv", sep='\t')
    for c in ['MD', 'INC']: data[c] = pd.to_numeric(data[c], errors='coerce')
    return data.dropna(subset=['MD'])

df = load_data()

# Modelo Multicapa WFRD
layers = [
    {"name": "Overburden", "tst": 20, "rh": 20, "rv": 30, "color": "#4b2c20"},
    {"name": "Upper Seal", "tst": 10, "rh": 2, "rv": 5, "color": "#757575"},
    {"name": "TARGET RES", "tst": 35, "rh": 150, "rv": 220, "color": "#FFD700"},
    {"name": "Basal / OWC", "tst": 50, "rh": 40, "rv": 70, "color": "#00008B"}
]

# --- SIDEBAR TÉCNICO ---
with st.sidebar:
    st.header("🎛️ WFRD Geosteering Control")
    st.session_state.dip = st.slider("DIP (°)", -15.0, 15.0, float(st.session_state.dip), step=0.1)
    st.session_state.shift = st.slider("Shift (ft)", -60.0, 60.0, float(st.session_state.shift), step=0.5)
    
    iter_type = st.radio("Intensidad de Inversión", [100, 1000])
    paleta_sel = st.selectbox("Paleta de Color", list(get_palettes().keys()))
    
    if st.button("🚀 Ejecutar Inversión Estocástica"):
        res_col = [c for c in df.columns if 'RAD' in c or 'RES' in c][0]
        b_s, b_d = run_proactive_inversion(df[res_col].tail(30).values, df['INC'].tail(30).values, layers, iterations=iter_type)
        st.session_state.shift, st.session_state.dip = float(b_s), float(b_d)
        st.session_state.history.append({"MD": df['MD'].iloc[-1], "DIP": b_d})

# --- TRACKS DE GEONAVEGACIÓN ---
st.title("WFRD Proactive Multilayer Dashboard")

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.3, 0.7])

# TRACK 1: LOGS (Real vs 3D)
res_col = [c for c in df.columns if 'RAD' in c or 'RES' in c][0]
fig.add_trace(go.Scatter(x=df['MD'], y=df[res_col], name="Log Real", line=dict(color='white')), row=1, col=1)
synth_3d = [calculate_3d_horns(layers[2]['rh'], layers[2]['rv'], inc, st.session_state.dip, st.session_state.shift) for inc in df['INC']]
fig.add_trace(go.Scatter(x=df['MD'], y=synth_3d, name="Modelo 3D", line=dict(color='red', dash='dot')), row=1, col=1)

# TRACK 2: CORTINA MULTICAPA + PROYECCIÓN 200FT
md_vals = df['MD'].values
y_base = -5160.0 + st.session_state.shift

for ly in layers:
    # Capas siguiendo la estructura del DIP
    y_layer = y_base + (md_vals * np.tan(np.radians(st.session_state.dip)))
    fig.add_trace(go.Scatter(x=md_vals, y=y_layer, fill='toself', fillcolor=ly['color'], line_width=0, name=ly['name']), row=2, col=1)
    
    # PROYECCIÓN LOOK-AHEAD 200FT
    md_ahead = np.linspace(md_vals[-1], md_vals[-1] + 200, 15)
    y_ahead = y_layer[-1] + ((md_ahead - md_vals[-1]) * np.tan(np.radians(st.session_state.dip)))
    fig.add_trace(go.Scatter(x=md_ahead, y=y_ahead, line=dict(color=ly['color'], dash='dash'), showlegend=False), row=2, col=1)
    
    y_base -= ly['tst']

# Trayectoria
fig.add_trace(go.Scatter(x=md_vals, y=np.full_like(md_vals, -5160), name="Pozo", line=dict(color='lime', width=3)), row=2, col=1)

fig.update_layout(height=900, template="plotly_dark")
fig.update_yaxes(type="log", row=1, col=1)
fig.update_yaxes(autorange="reversed", row=2, col=1)
st.plotly_chart(fig, use_container_width=True)

# MÉTRICAS PROACTIVAS
tvdss, tvt, dtbss = get_geo_metrics(md_vals[-1], df['INC'].iloc[-1], st.session_state.dip, st.session_state.shift, layers[2]['tst'])
c1, c2, c3, c4 = st.columns(4)
c1.metric("DTBss (Top)", f"{dtbss:.1f} ft")
c2.metric("TST Reservorio", f"{layers[2]['tst']} ft")
c3.metric("TVT Espesor", f"{tvt:.1f} ft")
c4.metric("DIP Calculado", f"{st.session_state.dip:.1f}°")
