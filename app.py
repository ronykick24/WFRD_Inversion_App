import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from engine_wfrd import run_ahta_inversion
from physics_engine import calculate_tst

st.set_page_config(layout="wide", page_title="WFRD Geosteering")

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
layers = [
    {"name": "Sello", "tst": 15, "rh": 4, "rv": 8, "color": "#4b2c20"},
    {"name": "Reservorio", "tst": 35, "rh": 180, "rv": 250, "color": "#FFD700"},
    {"name": "Basal", "tst": 45, "rh": 12, "rv": 20, "color": "#00008B"}
]

with st.sidebar:
    st.header("⚙️ Controles")
    st.session_state.dip = st.slider("DIP (°)", -15.0, 15.0, float(st.session_state.dip), 0.1)
    st.session_state.shift = st.slider("DTBss (ft)", -60.0, 60.0, float(st.session_state.shift), 0.5)
    
    if not df.empty and st.button("🚀 Ejecutar Inversión"):
        res_col = [c for c in df.columns if 'RAD' in c or 'RES' in c][0]
        res = run_ahta_inversion(df[res_col].tail(40), df['INC'].tail(40), layers)
        st.session_state.shift, st.session_state.dip = res[0], res[1]

c1, c2 = st.columns([0.7, 0.3])

with c1:
    st.subheader("Cortina Estructural")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.3, 0.7])
    
    md = df['MD'].values.tolist()
    z_mesh = np.linspace(-55, 55, 110).tolist()
    grid = [] # Construcción de grid para Heatmap

    for z in z_mesh:
        row = []
        for x in md:
            l_shift = st.session_state.shift + (x * np.tan(np.radians(st.session_state.dip)))
            rel_z = z - l_shift
            val = layers[1]['rh'] if -35 < rel_z < 0 else layers[0]['rh']
            row.append(val)
        grid.append(row)

    fig.add_trace(go.Heatmap(x=md, y=z_mesh, z=grid, colorscale='Cividis', showscale=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=md, y=[0]*len(md), name="Pozo", line=dict(color='lime', width=3)), row=2, col=1)
    
    res_c = [c for c in df.columns if 'RAD' in c or 'RES' in c][0]
    fig.add_trace(go.Scatter(x=md, y=df[res_c].tolist(), name="Log", line=dict(color='white')), row=1, col=1)
    
    fig.update_layout(height=800, template="plotly_dark", showlegend=False)
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Bin Plot (16 Sectores)")
    
    # --- FIX FINAL PARA VALUEERROR ---
    # 1. Crear listas nativas de Python con precisión controlada
    theta_py = [0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5, 180.0, 202.5, 225.0, 247.5, 270.0, 292.5, 315.0, 337.5]
    
    # 2. Calcular 'r' asegurando que no haya NaNs o tipos extraños
    s = float(st.session_state.shift)
    r_py = [float(np.exp(-abs(s)/15.0) * abs(np.cos(np.radians(t)))) for t in theta_py]
    
    # 3. Construir el objeto gráfico con validación explícita
    fig_polar = go.Figure(data=[
        go.Barpolar(
            r=r_py,
            theta=theta_py,
            marker_color=r_py,
            marker_colorscale='Viridis'
        )
    ])
    
    fig_polar.update_layout(
        template="plotly_dark", height=400,
        polar=dict(radialaxis=dict(visible=False), angularaxis=dict(direction="clockwise", rotation=90))
    )
    st.plotly_chart(fig_polar, use_container_width=True)
    
    tst_val = calculate_tst(layers[1]['tst'], st.session_state.dip)
    st.metric("DTBss", f"{st.session_state.shift:.2f} ft")
    st.metric("DIP", f"{st.session_state.dip:.2f} °")
    st.success(f"**TST Arena:** {tst_val:.1f} ft")
