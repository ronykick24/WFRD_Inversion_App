import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from engine_wfrd import run_ahta_inversion
from physics_engine import calculate_tst

st.set_page_config(layout="wide", page_title="WFRD AHTA Dashboard")

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
    {"name": "Transición", "tst": 12, "rh": 2, "rv": 4, "color": "#757575"},
    {"name": "TARGET", "tst": 35, "rh": 180, "rv": 250, "color": "#FFD700"},
    {"name": "Basal/OWC", "tst": 45, "rh": 12, "rv": 20, "color": "#00008B"}
]

with st.sidebar:
    st.header("⚙️ Geonavegación")
    st.session_state.dip = st.slider("DIP (°)", -15.0, 15.0, float(st.session_state.dip), 0.1)
    st.session_state.shift = st.slider("DTBss (ft)", -60.0, 60.0, float(st.session_state.shift), 0.5)
    
    if not df.empty and st.button("🚀 Ejecutar Inversión"):
        res_col = [c for c in df.columns if 'RAD' in c or 'RES' in c][0]
        b_s, b_d = run_ahta_inversion(df[res_col].tail(40), df['INC'].tail(40), layers)
        st.session_state.shift, st.session_state.dip = float(b_s), float(b_d)

c_main, c_bin = st.columns([0.7, 0.3])

with c_main:
    st.subheader("Sección Estructural (Curtain View)")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.3, 0.7])
    
    md = df['MD'].values
    z_mesh = np.linspace(-55, 55, 110)
    grid = np.zeros((len(z_mesh), len(md)))
    dip_rad = np.radians(st.session_state.dip)

    for i, x in enumerate(md):
        l_shift = st.session_state.shift + (x * np.tan(dip_rad))
        for j, z in enumerate(z_mesh):
            rel_z = z - l_shift
            cum, val = 0, layers[-1]['rh']
            for ly in layers:
                if rel_z >= -cum - ly['tst'] and rel_z <= -cum:
                    val = ly['rh']; break
                cum += ly['tst']
            grid[j, i] = val

    fig.add_trace(go.Heatmap(x=md, y=z_mesh, z=grid, colorscale='Cividis', showscale=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=md, y=np.zeros_like(md), name="Pozo", line=dict(color='lime', width=3)), row=2, col=1)
    
    res_c = [c for c in df.columns if 'RAD' in c or 'RES' in c][0]
    fig.add_trace(go.Scatter(x=md, y=df[res_c], name="Log", line=dict(color='white')), row=1, col=1)
    
    fig.update_layout(height=800, template="plotly_dark", showlegend=False)
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(autorange="reversed", title="Vertical Offset (ft)", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

with c_bin:
    st.subheader("Bin Plot (16 Sectores)")
    
    # SOLUCIÓN AL VALUEERROR:
    # 1. Definir ángulos como lista de floats
    # 2. Definir r como lista de floats
    theta_list = [float(x) for x in np.linspace(0, 360, 16, endpoint=False)]
    
    # Intensidad basada en DTBss (0° es el Highside del pozo)
    val_base = np.exp(-abs(st.session_state.shift)/15.0)
    r_list = [float(val_base * abs(np.cos(np.radians(t)))) for t in theta_list]
    
    fig_polar = go.Figure()
    fig_polar.add_trace(go.Barpolar(
        r=r_list,
        theta=theta_list,
        marker_color=r_list,
        colorscale='Viridis'
    ))
    fig_polar.update_layout(
        template="plotly_dark", 
        height=400,
        polar=dict(
            radialaxis=dict(visible=False),
            angularaxis=dict(direction="clockwise", rotation=90)
        )
    )
    st.plotly_chart(fig_polar, use_container_width=True)
    
    # Métricas Proactivas
    tst_actual = calculate_tst(layers[2]['tst'], st.session_state.dip)
    st.metric("DTBss (Boundary)", f"{st.session_state.shift:.2f} ft")
    st.metric("DIP Estructural", f"{st.session_state.dip:.2f} °")
    st.success(f"**TST de Reservorio:** {tst_actual:.1f} ft")
