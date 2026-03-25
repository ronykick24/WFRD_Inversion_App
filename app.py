import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from engine_wfrd import run_ahta_inversion
from physics_engine import calculate_tst

st.set_page_config(layout="wide", page_title="WFRD Geosteering Dashboard")

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
    {"name": "Sello", "tst": 15, "rh": 4, "rv": 8},
    {"name": "Reservorio", "tst": 35, "rh": 180, "rv": 250},
    {"name": "Basal", "tst": 45, "rh": 12, "rv": 20}
]

with st.sidebar:
    st.header("⚙️ Geonavegación")
    st.session_state.dip = st.slider("DIP (°)", -15.0, 15.0, float(st.session_state.dip), 0.1)
    st.session_state.shift = st.slider("DTBss (ft)", -60.0, 60.0, float(st.session_state.shift), 0.5)
    
    if not df.empty and st.button("🚀 Ejecutar Inversión AHTA"):
        res_col = [c for c in df.columns if 'RAD' in c or 'RES' in c][0]
        sol = run_ahta_inversion(df[res_col].tail(40), df['INC'].tail(40), layers)
        st.session_state.shift, st.session_state.dip = sol[0], sol[1]

c1, c2 = st.columns([0.7, 0.3])

with c1:
    st.subheader("Cortina Estructural (Proactiva)")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.3, 0.7])
    
    md_list = df['MD'].values.tolist()
    z_mesh = np.linspace(-55, 55, 110).tolist()
    grid = []

    for z in z_mesh:
        row = []
        for x in md_list:
            l_shift = st.session_state.shift + (x * np.tan(np.radians(st.session_state.dip)))
            rel_z = z - l_shift
            val = layers[1]['rh'] if -35 < rel_z < 0 else layers[0]['rh']
            row.append(val)
        grid.append(row)

    fig.add_trace(go.Heatmap(x=md_list, y=z_mesh, z=grid, colorscale='Cividis', showscale=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=md_list, y=[0]*len(md_list), name="Pozo", line=dict(color='lime', width=3)), row=2, col=1)
    
    res_c = [c for c in df.columns if 'RAD' in c or 'RES' in c][0]
    fig.add_trace(go.Scatter(x=md_list, y=df[res_c].tolist(), name="Log", line=dict(color='white')), row=1, col=1)
    
    fig.update_layout(height=800, template="plotly_dark", showlegend=False)
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Bin Plot (16 Sectores)")
    
    # --- SOLUCIÓN DEFINITIVA AL VALUEERROR ---
    # Forzar listas nativas de Python para evitar errores de validación de Plotly
    theta_angles = [float(x) for x in [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5]]
    
    s_val = float(st.session_state.shift)
    # Cálculo de r asegurando que sea una lista de floats puros
    r_values = [float(np.exp(-abs(s_val)/15.0) * abs(np.cos(np.radians(a)))) for a in theta_angles]
    
    # Crear el gráfico usando la sintaxis de lista de trazas (más robusta)
    fig_bin = go.Figure(data=[
        go.Barpolar(
            r=r_values,
            theta=theta_angles,
            marker_color=r_values,
            marker_colorscale='Viridis'
        )
    ])
    
    fig_bin.update_layout(
        template="plotly_dark", height=400,
        polar=dict(
            radialaxis=dict(visible=False),
            angularaxis=dict(direction="clockwise", rotation=90, tickvals=theta_angles)
        )
    )
    st.plotly_chart(fig_bin, use_container_width=True)
    
    tst_actual = calculate_tst(layers[1]['tst'], st.session_state.dip)
    st.metric("DTBss", f"{st.session_state.shift:.2f} ft")
    st.metric("DIP", f"{st.session_state.dip:.2f} °")
    st.success(f"**TST Reservorio:** {tst_actual:.1f} ft")
