import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# CORRECCIÓN DE IMPORTACIÓN: Debe coincidir con el nombre del archivo en GitHub
from engine_wfrd import run_proactive_inversion 
from physics_engine import calculate_3d_horns, get_geo_metrics
from utils import save_geosteering_report

st.set_page_config(layout="wide", page_title="WFRD Master Suite")

# --- PERSISTENCIA ---
if 'shift' not in st.session_state: st.session_state.shift = 0.0
if 'dip' not in st.session_state: st.session_state.dip = 0.0
if 'ghost' not in st.session_state: st.session_state.ghost = {"s": 0.0, "d": 0.0}

# Carga de datos reales desde tu archivo .tsv
@st.cache_data
def load_data():
    try:
        return pd.read_csv("well_logs.tsv", sep='\t')
    except:
        return pd.DataFrame()

df = load_data()

# Capas (Modelo basado en tu imagen)
layers = [
    {"name": "Overburden", "tst": 20, "rh": 20, "rv": 30, "color": "#4b2c20"},
    {"name": "Upper Seal", "tst": 5, "rh": 2, "rv": 3, "color": "#757575"},
    {"name": "TARGET RESERVOIR", "tst": 25, "rh": 100, "rv": 150, "color": "#FFD700"},
    {"name": "Basal Sand (OWC)", "tst": 40, "rh": 50, "rv": 70, "color": "#00008B"}
]

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ WFRD Controls")
    st.session_state.dip = st.slider("DIP (°)", -20.0, 20.0, st.session_state.dip)
    st.session_state.shift = st.slider("Shift (ft)", -60.0, 60.0, st.session_state.shift)
    
    if not df.empty and st.button("🚀 Ejecutar Inversión Estocástica"):
        last_data = df.tail(10)
        # Suponiendo que la columna de resistividad se llama 'RAD1_GW6'
        res_col = [c for c in df.columns if 'RAD' in c][0] 
        b_s, b_d = run_proactive_inversion(last_data[res_col].values, last_data['MD'].values, last_data['INC'].values, layers)
        st.session_state.shift, st.session_state.dip = b_s, b_d
    
    st.button("💾 Fijar Ghost", on_click=lambda: st.session_state.update(ghost={"s": st.session_state.shift, "d": st.session_state.dip}))

# --- VISUALIZACIÓN ---
st.title("Weatherford GuideWave Master Dashboard")

if df.empty:
    st.warning("No se encontraron datos en well_logs.tsv")
else:
    md = df['MD'].values
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.2, 0.2, 0.6])

    # Track de Logs (Datos Reales)
    res_col = [c for c in df.columns if 'RAD' in c][0]
    fig.add_trace(go.Scatter(x=md, y=df[res_col], name="Real"), row=1, col=1)
    
    # Track de Cortina
    y_base = -5160 + st.session_state.shift
    for ly in layers:
        y_layer = y_base + (md * np.tan(np.radians(st.session_state.dip)))
        fig.add_trace(go.Scatter(x=md, y=y_layer, name=ly['name'], fill='toself', fillcolor=ly['color'], line_width=0), row=3, col=1)
        fig.add_annotation(x=md[-1]+50, y=y_layer[-1], text=f"{ly['rh']} ohm", showarrow=False, row=3, col=1)
        y_base -= ly['tst']

    fig.update_layout(height=850, template="plotly_dark", showlegend=False)
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=3, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # Métricas
    tvdss, tvt, dtb = get_geo_metrics(md[-1], df['INC'].iloc[-1], st.session_state.dip, st.session_state.shift, 25)
    st.columns(3)[0].metric("TVDss", f"{tvdss:.1f}")
    st.columns(3)[1].metric("DTBss (Techo)", f"{abs(dtb):.1f}")
    st.columns(3)[2].metric("TVT", f"{tvt:.1f}")
