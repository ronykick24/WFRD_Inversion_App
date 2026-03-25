import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Imports desde tus archivos de GitHub
from engine_wfrd import run_proactive_inversion 
from physics_engine import calculate_3d_horns, get_geo_metrics
from utils import save_geosteering_report

st.set_page_config(layout="wide", page_title="WFRD Suite")

# --- PERSISTENCIA ---
if 'shift' not in st.session_state: st.session_state.shift = 0.0
if 'dip' not in st.session_state: st.session_state.dip = 0.0
if 'ghost' not in st.session_state: st.session_state.ghost = {"s": 0.0, "d": 0.0}

# Carga de datos reales
@st.cache_data
def load_data():
    try:
        # Cargamos tu archivo well_logs.tsv
        data = pd.read_csv("well_logs.tsv", sep='\t')
        return data.dropna(subset=['MD', 'INC'])
    except:
        return pd.DataFrame()

df = load_data()

# Modelo de Capas
layers = [
    {"name": "Overburden", "tst": 20, "rh": 20, "rv": 30, "color": "#4b2c20"},
    {"name": "Upper Seal", "tst": 5, "rh": 2, "rv": 3, "color": "#757575"},
    {"name": "TARGET RESERVOIR", "tst": 25, "rh": 100, "rv": 150, "color": "#FFD700"},
    {"name": "Basal Sand (OWC)", "tst": 40, "rh": 50, "rv": 70, "color": "#00008B"}
]

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Geosteering Panel")
    st.session_state.dip = st.slider("DIP (°)", -20.0, 20.0, st.session_state.dip)
    st.session_state.shift = st.slider("Shift (ft)", -60.0, 60.0, st.session_state.shift)
    
    if not df.empty and st.button("🚀 Run Inversion"):
        # Usamos los últimos 15 puntos para estabilizar el cálculo
        window = df.tail(15)
        # Buscamos automáticamente la columna de resistividad
        res_col = [c for c in df.columns if 'RAD' in c or 'RES' in c][0]
        
        # LLAMADA CORREGIDA PARA EVITAR TYPEERROR
        best_s, best_d = run_proactive_inversion(
            window[res_col].values, 
            window['INC'].values, 
            layers
        )
        st.session_state.shift, st.session_state.dip = float(best_s), float(best_d)
    
    st.button("💾 Save Ghost", on_click=lambda: st.session_state.update(ghost={"s": st.session_state.shift, "d": st.session_state.dip}))

# --- DASHBOARD ---
st.title("WFRD Proactive Dashboard")

if df.empty:
    st.error("No se detectó el archivo well_logs.tsv en el repositorio.")
else:
    # Gráficos de Plotly
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.3, 0.7])
    
    # Logs Reales
    res_col = [c for c in df.columns if 'RAD' in c or 'RES' in c][0]
    fig.add_trace(go.Scatter(x=df['MD'], y=df[res_col], name="Log Real", line=dict(color='white')), row=1, col=1)
    
    # Cortina Estructural
    y_base = -5160 + st.session_state.shift
    for ly in layers:
        y_layer = y_base + (df['MD'] * np.tan(np.radians(st.session_state.dip)))
        fig.add_trace(go.Scatter(x=df['MD'], y=y_layer, fill='toself', fillcolor=ly['color'], line_width=0, name=ly['name']), row=2, col=1)
        y_base -= ly['tst']

    fig.update_layout(height=800, template="plotly_dark", showlegend=False)
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # Métricas finales
    tvdss, tvt, dtb = get_geo_metrics(df['MD'].iloc[-1], df['INC'].iloc[-1], st.session_state.dip, st.session_state.shift, 25)
    st.metric("DTBss (Techo)", f"{abs(dtb):.1f} ft")
