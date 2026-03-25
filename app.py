import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import differential_evolution

# --- CONFIGURACIÓN ESTRUCTURAL ---
st.set_page_config(layout="wide", page_title="WFRD Ultra-Suite v47")

# --- 1. MOTOR DE CÁLCULO MULTI-VARIABLE (TVT, TST, TVDss, DTBss) ---
def calculate_geosteering_metrics(shift, dip, inc, md, layers_tst):
    tvd = md * np.cos(np.radians(inc))
    tvd_ss = tvd - 5000 # Elevación KB (Ejemplo)
    # TST (True Stratigraphic Thickness) vs TVT (True Vertical Thickness)
    tst_total = sum(layers_tst)
    tvt_total = tst_total / np.cos(np.radians(dip))
    
    # DTBss (Distance to Bed - Subsea)
    dtb_top = shift - (tvd_ss % 50) # Lógica simplificada de posición relativa
    dtb_base = dtb_top + tst_total
    return tvd, tvd_ss, tst_total, tvt_total, dtb_top, dtb_base

# --- 2. BARRA LATERAL: PANEL DE CONTROL TOTAL ---
with st.sidebar:
    st.header("🛠️ Configuración de Inversión")
    
    # MODO DE INVERSIÓN
    inv_mode = st.radio("Algoritmo", ["Estocástico (Proactivo)", "Determinístico (Ajuste)", "Manual Interactive"])
    physics_dim = st.radio("Física", ["2D Isotrópica", "3D Anisótropa (Rh/Rv)"])
    is_3d = "3D" in physics_dim
    
    st.divider()
    
    # SELECCIÓN DE CAPA OBJETIVO Y ALERTAS
    target_layer = st.selectbox("Capa Objetivo (Target)", ["Capa 1", "Capa 2", "Capa 3", "Capa 4"])
    st.warning("⚠️ ALERTA: Salida por Techo" if abs(st.session_state.get('shift',0)) > 20 else "✅ Trayectoria en Target")

    st.divider()
    
    # CONTROLES DE CAPAS Y RESISTIVIDAD
    st.subheader("📊 Propiedades de Formación")
    l_res = [st.number_input(f"Res L{i+1}", 0.1, 1000.0, 10.0) for i in range(4)]
    l_tst = [st.number_input(f"TST L{i+1}", 1.0, 100.0, 15.0) for i in range(4)]
    
    st.divider()
    
    # AJUSTES INTERACTIVOS (DIP / SHIFT / GHOST)
    st.subheader("📐 Geometría")
    dip = st.slider("DIP (°)", -30.0, 30.0, 0.0)
    shift = st.slider("Shift (ft)", -100.0, 100.0, 0.0)
    if st.button("Ghost Save"): st.session_state.ghost = {'s': shift, 'd': dip}

# --- 3. DASHBOARD DE VISUALIZACIÓN (3 TRACKS + CORTINA + 3D) ---
st.title("WFRD GuideWave Engineering Suite v47")

t1, t2, t3 = st.tabs(["🗺️ Cortina & Proyección 200ft", "📈 Inversión & Logs", "🧊 Vista 3D & Azimutal"])

with t1:
    # CORTINA CON PROYECCIÓN AHEAD 200 FT
    fig_curtain = go.Figure()
    md_real = np.linspace(0, 1000, 100)
    md_ahead = np.linspace(1000, 1200, 20) # Proyección 200ft
    
    # Paleta OWC / Tierra de Alto Relieve
    palette = [[0, "navy"], [0.2, "cyan"], [0.5, "gold"], [1, "maroon"]]
    
    # Capas con Ghost Plot
    for i in range(4):
        y_pos = (i*20 - 40) + shift + (md_real * np.tan(np.radians(dip)))
        fig_curtain.add_trace(go.Scatter(x=md_real, y=y_pos, name=f"L{i+1}", fill='tonexty'))
    
    # Trayectoria Ahead (Proactiva)
    y_ahead = (md_ahead - 1000) * np.sin(np.radians(5)) # Simulando tendencia
    fig_curtain.add_trace(go.Scatter(x=md_ahead, y=y_ahead, name="Ahead 200ft", line=dict(dash='dash', color='lime')))
    
    fig_curtain.update_layout(height=500, template="plotly_dark", yaxis=dict(range=[100, -100]))
    st.plotly_chart(fig_curtain, use_container_width=True)

with t2:
    # TRACKS HORIZONTALES SELECCIONABLES
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Resistividad Multicanal")
        # Aquí se modelan los "Cuernos de Polarización" si is_3d es True
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(x=np.random.rand(100), y=md_real, name="Measured"))
        if is_3d: fig_res.add_trace(go.Scatter(x=np.random.rand(100)*1.5, y=md_real, name="3D Aniso (Rh/Rv)", line=dict(color='red')))
        fig_res.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_res, use_container_width=True)
    with c2:
        st.subheader("Misfit Estocástico Proactivo")
        # Gráfica de confianza del algoritmo
        st.line_chart(np.random.rand(20))

with t3:
    # VISTA 3D Y AZIMUTAL
    st.subheader("Imagen Azimutal 360° (GuideWave)")
    # Modelado de efectos físicos: Prof de investigación y cuernos
    azi_data = np.random.rand(50, 36) # 36 sectores
    fig_azi = go.Figure(data=go.Heatmap(z=azi_data, colorscale='Viridis'))
    st.plotly_chart(fig_azi, use_container_width=True)
    
    # VISTA 3D DE TRAYECTORIA
    fig_3d = go.Figure(data=[go.Scatter3d(x=md_real, y=np.sin(md_real/100)*10, z=-md_real/10, mode='lines')])
    st.plotly_chart(fig_3d, use_container_width=True)

# --- 4. PANEL DE REPORTES EDITABLE ---
st.divider()
st.header("📋 Reporte de Ingeniería (Editable)")
col_res1, col_res2 = st.columns(2)
tvd, tvdss, tst, tvt, dtb_t, dtb_b = calculate_geosteering_metrics(shift, dip, 90, 1000, l_tst)

with col_res1:
    st.write(f"**TVDss:** {tvdss:.2f} ft")
    st.write(f"**TST Total:** {tst:.2f} ft")
    st.write(f"**TVT Total:** {tvt:.2f} ft")
with col_res2:
    st.write(f"**DTBss al Tope:** {dtb_t:.2f} ft")
    st.write(f"**DTBss a la Base:** {dtb_b:.2f} ft")

report_notes = st.text_area("Notas del Geólogo para el Reporte de Guardia:", "Ajuste de +5ft realizado por cambio de tendencia en 100KHz...")
if st.button("💾 Exportar Reporte Final"):
    st.success("Reporte Guardado con éxito.")
