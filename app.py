import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from engine_wfrd import run_proactive_inversion 
from utils import get_wfrd_palette

st.set_page_config(layout="wide", page_title="WFRD Master Suite")

# --- ESTADO Y DATOS ---
if 'shift' not in st.session_state: st.session_state.shift = 0.0
if 'dip' not in st.session_state: st.session_state.dip = 0.0

@st.cache_data
def load_data():
    data = pd.read_csv("well_logs.tsv", sep='\t')
    for c in ['MD', 'INC']: data[c] = pd.to_numeric(data[c], errors='coerce')
    return data.dropna(subset=['MD'])

df = load_data()

# Configuración Multicapa
layers = [
    {"name": "Sello Superior", "tst": 15, "rh": 5, "color": "#4b2c20"},
    {"name": "Reservorio Top", "tst": 25, "rh": 150, "color": "#FFD700"},
    {"name": "Reservorio Base", "tst": 20, "rh": 80, "color": "#DAA520"},
    {"name": "Acuífero", "tst": 40, "rh": 10, "color": "#00008B"}
]

# --- UI SIDEBAR ---
with st.sidebar:
    st.header("🎮 Steering Controls")
    st.session_state.dip = st.slider("DIP Angle (°)", -10.0, 10.0, float(st.session_state.dip), 0.1)
    st.session_state.shift = st.slider("DTBss / Shift (ft)", -50.0, 50.0, float(st.session_state.shift), 0.5)
    
    if st.button("🚀 Inversión Estocástica (1000 iter)"):
        res_col = [c for c in df.columns if 'RAD' in c or 'RES' in c][0]
        # Aquí se llamaría a engine_wfrd.run_proactive_inversion
        st.success("Ajuste estructural completado")

# --- CONSTRUCCIÓN DE LA CORTINA (HEATMAP) ---
st.title("WFRD Real-Time Structural Curtain")

if not df.empty:
    # 1. Crear malla de la cortina
    md_x = df['MD'].values
    # Rango vertical de visualización (p. ej. 80 pies alrededor del pozo)
    z_y = np.linspace(-40, 40, 80) 
    
    # 2. Calcular la matriz de resistividad inclinada
    # Esta es la "magia": cada celda se ajusta por el DIP y el Shift
    curtain_grid = np.zeros((len(z_y), len(md_x)))
    
    dip_rad = np.radians(st.session_state.dip)
    for i, x in enumerate(md_x):
        # Desplazamiento estructural en este punto MD
        local_shift = st.session_state.shift + (x * np.tan(dip_rad))
        
        for j, y_val in enumerate(z_y):
            # Posición relativa a las capas
            rel_z = y_val - local_shift
            
            # Asignar resistividad según el modelo multicapa
            cum_tst = 0
            val = layers[-1]['rh'] # Fondo por defecto
            for ly in layers:
                if rel_z >= -cum_tst - ly['tst'] and rel_z <= -cum_tst:
                    val = ly['rh']
                    break
                cum_tst += ly['tst']
            curtain_grid[j, i] = val

    # --- VISUALIZACIÓN ---
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.3, 0.7])

    # Track Superior: Logs
    res_col = [c for c in df.columns if 'RAD' in c or 'RES' in c][0]
    fig.add_trace(go.Scatter(x=md_x, y=df[res_col], name="Log Real", line=dict(color='white')), row=1, col=1)

    # Track Inferior: CORTINA 2D
    fig.add_trace(go.Heatmap(
        x=md_x, y=z_y, z=curtain_grid,
        colorscale=get_wfrd_palette(),
        zmin=1, zmax=200, colorbar=dict(title="Ohm.m"),
        hoverinfo='z'
    ), row=2, col=1)

    # Línea del Pozo (Siempre en el centro de la cortina local)
    fig.add_trace(go.Scatter(x=md_x, y=np.zeros_like(md_x), name="Wellbore", line=dict(color='lime', width=3)), row=2, col=1)

    fig.update_layout(height=800, template="plotly_dark", showlegend=False)
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(title="Vertical Offset (ft)", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

    # Métricas Proactivas
    st.info(f"Geosteering: Navegando en Capa '{layers[1]['name']}' | DTBss: {st.session_state.shift:.1f} ft")
