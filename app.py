import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# IMPORTACIÓN CON NOMBRES EXACTOS DE TU GITHUB
from engine_wfrd import run_proactive_inversion 
from physics_engine import calculate_3d_horns, get_geo_metrics
from utils import save_geosteering_report

st.set_page_config(layout="wide", page_title="WFRD Real-Time Suite")

# --- PERSISTENCIA ---
if 'shift' not in st.session_state: st.session_state.shift = 0.0
if 'dip' not in st.session_state: st.session_state.dip = 0.0
if 'ghost' not in st.session_state: st.session_state.ghost = {"s": 0.0, "d": 0.0}
if 'log_curve' not in st.session_state: st.session_state.log_curve = 'RAD1_GW6'

# Capas basadas en tu imagen de referencia
layers = [
    {"name": "Overburden", "tst": 20, "rh": 20, "rv": 30, "color": "#4b2c20"},
    {"name": "Upper Seal", "tst": 5, "rh": 2, "rv": 3, "color": "#757575"},
    {"name": "TARGET RESERVOIR", "tst": 25, "rh": 100, "rv": 150, "color": "#FFD700"},
    {"name": "Basal Sand (OWC)", "tst": 40, "rh": 50, "rv": 70, "color": "#00008B"}
]

# --- 1. CARGA DE DATOS REALES (well_logs.tsv) ---
@st.cache_data
def load_wfrd_logs():
    try:
        df = pd.read_csv("well_logs.tsv", sep='\t')
        # Limpieza básica de datos nulos si existen
        df = df.dropna(subset=['MD', 'INC'])
        return df
    except FileNotFoundError:
        return pd.DataFrame()

well_df = load_wfrd_logs()

# --- 2. SIDEBAR ACTUALIZADO: CONTROLES DE INGENIERÍA ---
with st.sidebar:
    st.header("⚙️ WFRD Controls")
    
    # Estatus de Carga de Datos
    if well_df.empty:
        st.error("🚨 Archivo well_logs.tsv no encontrado.")
        st.stop()
    else:
        st.success(f"✅ Datos cargados: {well_df.shape[0]} puntos.")
    
    st.divider()
    
    # Selector de Curva de Resistividad Real (dinámico)
    res_curves = [c for c in well_df.columns if 'RAD' in c or 'RPD' in c or 'GW6' in c]
    st.session_state.log_curve = st.selectbox("Curva de Inversión WFRD", res_curves, index=0)
    
    st.divider()
    
    # Sliders de DIP/Shift
    st.session_state.dip = st.slider("DIP (°)", -20.0, 20.0, st.session_state.dip)
    st.session_state.shift = st.slider("Shift (ft)", -60.0, 60.0, st.session_state.shift)
    
    # Botón para correr la inversión de engine_wfrd.py usando datos reales
    if st.button("🚀 Ejecutar Inversión Estocástica"):
        # Tomamos los últimos 20 puntos de datos para la inversión (Proactivo)
        inversion_window = well_df.tail(20)
        best_s, best_d = run_proactive_inversion(
            inversion_window[st.session_state.log_curve].values,
            inversion_window['MD'].values,
            inversion_window['INC'].values,
            layers
        )
        st.session_state.shift, st.session_state.dip = best_s, best_d
    
    if st.button("💾 Fijar Ghost (Sombra)"):
        st.session_state.ghost = {"s": st.session_state.shift, "d": st.session_state.dip}

# --- 3. DASHBOARD UNIFICADO: TRACKS APILADOS (DATOS REALES) ---
st.title("Weatherford GuideWave Real-Time Dashboard")

# Tracks compartiendo eje MD (como en tu imagen)
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.01,
                    row_heights=[0.25, 0.15, 0.6],
                    subplot_titles=("Logs LWD (RADs GW6)", "Misfit & QC", "Cortina & Proyección 200ft Ahead"))

# Track 1: Resistividad Real vs Sintética 3D
fig.add_trace(go.Scatter(x=well_df['MD'], y=well_df[st.session_state.log_curve], name=f"Real {st.session_state.log_curve}", line=dict(color='white')), row=1, col=1)
# Sintética 3D: Simulada en el bit con la física de physics_engine.py
synth_3d = []
for i in range(len(well_df)):
    rh, rv = layers[2]['rh'], layers[2]['rv'] # Capa Reservorio
    horn = calculate_3d_horns(rh, rv, well_df['INC'].iloc[i], st.session_state.dip, st.session_state.shift)
    synth_3d.append(horn)
fig.add_trace(go.Scatter(x=well_df['MD'], y=synth_3d, name="Synth 3D", line=dict(color='red', dash='dot')), row=1, col=1)

# Track 3: Cortina Geológica Técnica (con Ghost, OWC y Ahead)
y_base = -5160 + st.session_state.shift # TVD base de ejemplo
for ly in layers:
    # Dibuja la capa respetando el Shift y DIP estructural
    y_layer = y_base + (well_df['MD'] * np.tan(np.radians(st.session_state.dip)))
    fig.add_trace(go.Scatter(x=well_df['MD'], y=y_layer, name=ly['name'], fill='toself', fillcolor=ly['color'], line_width=0), row=3, col=1)
    
    # Etiqueta de resistividad a la derecha (replicado de tu imagen)
    fig.add_annotation(x=well_df['MD'].max() + 50, y=y_layer.iloc[-1], text=f"{ly['rh']} ohm", showarrow=False, row=3, col=1)
    y_base -= ly['tst'] # Calcular espesor para la siguiente capa

# GHOST PLOT (Sombra Structural del ajuste anterior)
if st.session_state.ghost['s'] != 0:
    ghost_y = -5160 + st.session_state.ghost['s'] + (well_df['MD'] * np.tan(np.radians(st.session_state.ghost['d'])))
    fig.add_trace(go.Scatter(x=well_df['MD'], y=ghost_y, name="Ghost Plan", line=dict(color='rgba(255,255,255,0.4)', dash='dot')), row=3, col=1)

# PROYECCIÓN PROACTIVA AHEAD 200FT
md_ahead = np.linspace(well_df['MD'].max(), well_df['MD'].max()+200, 20)
# Proyecta la trayectoria recta si el geólogo no ha intervenido
fig.add_trace(go.Scatter(x=md_ahead, y=np.full_like(md_ahead, -5160), name="Ahead 200ft", line=dict(color='lime', dash='dash', width=3)), row=3, col=1)

# --- CONFIGURACIÓN DE EJES Y ALERTAS DE INGENIERÍA ---
fig.update_layout(height=1000, template="plotly_dark", showlegend=False)
fig.update_xaxes(showgrid=True, gridcolor='gray')
fig.update_yaxes(type="log", title="Ω·m", row=1, col=1)
# Track geológico en TVD (autorange reversed para que 0 esté arriba)
fig.update_yaxes(autorange="reversed", title="TVD Perp (ft)", range=[-5100, -5250], row=3, col=1)

st.plotly_chart(fig, use_container_width=True)

# --- PANEL DE INGENIERÍA Y REPORTES ---
last_row = well_df.iloc[-1]
tvdss, tvt, dtbss_t, tvd_perp = get_geo_metrics(last_row['MD'], last_row['INC'], st.session_state.dip, st.session_state.shift, 25)

c1, c2, c3, c4 = st.columns(4)
c1.metric("TVDss", f"{tvdss:.1f} ft")
c2.metric("TVT Total", f"{tvt:.1f} ft")
c3.metric("DTBss (Techo)", f"{abs(dtbss_t):.1f} ft", delta="-0.3 (Acercándose)")
c4.metric("TVD Real", f"{tvd_perp:.1f} ft")

# Alerta de Salida de Reservorio (Capa Objetivo)
if abs(dtbss_t) < 5.0: # Simulación de alerta basada en DTBss
    st.error("🚨 CRÍTICO: ACERCÁNDOSE AL TECHO DEL RESERVORIO")

report_notes = st.text_area("Notas del Geólogo para el Reporte de Guardia:", "Inversión estocástica sugiere DIP estructural de +3.2°...")
if st.button("💾 Generar Reporte Editable (CSV)"):
    report_df = save_geosteering_report({"TVDss": tvdss, "DTBss": dtbss_t, "DIP": st.session_state.dip, "Notas": report_notes})
    st.success("Reporte compilado satisfactoriamente.")
    st.write(report_df)
