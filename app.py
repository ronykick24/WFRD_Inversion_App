import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Importación de tus módulos locales
from physics_engine import calculate_3d_horns, get_geo_metrics
from engine import run_proactive_inversion
from utils import get_owc_palette, save_geosteering_report

st.set_page_config(layout="wide", page_title="WFRD GuideWave Suite")

# --- PERSISTENCIA (No se borra al mover sliders) ---
if 'shift' not in st.session_state: st.session_state.shift = 0.0
if 'dip' not in st.session_state: st.session_state.dip = 0.0
if 'ghost' not in st.session_state: st.session_state.ghost = {"s": 0.0, "d": 0.0}

# --- MODELO DE CAPAS (Basado en imagen_4.png) ---
layers = [
    {"name": "Overburden", "tst": 20, "rh": 20, "rv": 30, "color": "#4b2c20"},
    {"name": "Upper Seal", "tst": 5, "rh": 2, "rv": 3, "color": "#757575"},
    {"name": "TARGET RESERVOIR", "tst": 25, "rh": 100, "rv": 150, "color": "#FFD700"},
    {"name": "Basal Sand (OWC)", "tst": 40, "rh": 50, "rv": 70, "color": "#00008B"}
]

# --- SIDEBAR: CONTROLES DE INGENIERÍA ---
with st.sidebar:
    st.header("⚙️ WFRD Controls")
    st.session_state.dip = st.slider("DIP (°)", -20.0, 20.0, st.session_state.dip)
    st.session_state.shift = st.slider("Shift (ft)", -60.0, 60.0, st.session_state.shift)
    
    if st.button("🚀 Ejecutar Inversión Estocástica"):
        b_s, b_d = run_proactive_inversion(105.0, 89.5, layers)
        st.session_state.shift, st.session_state.dip = b_s, b_d
    
    if st.button("💾 Fijar Ghost (Sombra)"):
        st.session_state.ghost = {"s": st.session_state.shift, "d": st.session_state.dip}

# --- TRACKS APILADOS (VISUALIZACIÓN TÉCNICA) ---
st.title("Weatherford Proactive Geosteering Master Dashboard")
md = np.linspace(1500, 3500, 100)

fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                    row_heights=[0.2, 0.2, 0.6],
                    subplot_titles=("Resistivity Logs (GW6)", "Misfit QC", "Curtain Plot & Ahead 200ft"))

# TRACK 1: Logs con Cuernos de Polarización
fig.add_trace(go.Scatter(x=md, y=np.random.lognormal(2, 0.1, 100), name="Real"), row=1, col=1)
fig.add_trace(go.Scatter(x=md, y=np.random.lognormal(2, 0.05, 100), name="Synth 3D", line=dict(color='red', dash='dot')), row=1, col=1)

# TRACK 3: Cortina Geológica + Ghost + Proyección
y_base = -5160 + st.session_state.shift
for ly in layers:
    y_layer = y_base + (md * np.tan(np.radians(st.session_state.dip)))
    fig.add_trace(go.Scatter(x=md, y=y_layer, name=ly['name'], fill='toself', fillcolor=ly['color'], line_width=0), row=3, col=1)
    # Etiqueta de resistividad (como en tu imagen)
    fig.add_annotation(x=3550, y=y_layer[-1], text=f"{ly['rh']} ohm", showarrow=False, row=3, col=1)
    y_base -= ly['tst']

# Línea Ghost (Sombra del ajuste anterior)
ghost_y = -5160 + st.session_state.ghost['s'] + (md * np.tan(np.radians(st.session_state.ghost['d'])))
fig.add_trace(go.Scatter(x=md, y=ghost_y, name="Ghost", line=dict(color='rgba(255,255,255,0.4)', dash='dot')), row=3, col=1)

# Proyección Ahead 200ft (Verde punteada)
md_ahead = np.linspace(3500, 3700, 10)
fig.add_trace(go.Scatter(x=md_ahead, y=np.full(10, -5160), name="Ahead", line=dict(color='lime', dash='dash', width=3)), row=3, col=1)

fig.update_layout(height=850, template="plotly_dark", showlegend=False)
fig.update_yaxes(type="log", row=1, col=1)
fig.update_yaxes(autorange="reversed", row=3, col=1)
st.plotly_chart(fig, use_container_width=True)

# --- PANEL DE MÉTRICAS Y REPORTES ---
tvdss, tvt, dtb_t, dtb_b = get_geo_metrics(3500, 89.5, st.session_state.dip, st.session_state.shift, 25)
c1, c2, c3, c4 = st.columns(4)
c1.metric("TVDss", f"{tvdss:.1f} ft")
c2.metric("TVT Total", f"{tvt:.1f} ft")
c3.metric("DTBss (Techo)", f"{abs(dtb_t):.1f} ft")
c4.metric("DIP Invertido", f"{st.session_state.dip}°")

if abs(dtb_t) < 5:
    st.error("🚨 ALERTA: Salida inminente por el TECHO del Reservorio")

# Reporte Editable
obs = st.text_area("Notas de Guardia:", "Ajuste proactivo realizado...")
if st.button("💾 Exportar Reporte CSV"):
    report_df = save_geosteering_report({"TVDss": tvdss, "DIP": st.session_state.dip, "Observaciones": obs})
    st.write(report_df)
