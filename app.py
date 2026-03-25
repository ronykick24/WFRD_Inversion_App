import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(layout="wide", page_title="WFRD Master Suite v45")

# --- ESTADO DE SESIÓN PARA PERSISTENCIA (SHIFT/DIP/GHOST) ---
if 'prev_shift' not in st.session_state: st.session_state.prev_shift = 0.0
if 'prev_dip' not in st.session_state: st.session_state.prev_dip = 0.0

# --- 1. BARRA LATERAL: TODOS LOS CONTROLES RESTAURADOS ---
with st.sidebar:
    st.header("🎮 Controles de Geosteering")
    
    # SECCIÓN DE CAPAS (LABELS)
    st.subheader("🏷️ Definición de Capas")
    n_layers = st.number_input("Número de Capas", 3, 7, 5)
    layer_names = []
    for i in range(int(n_layers)):
        name = st.text_input(f"Nombre Capa {i+1}", f"Capa {i+1}")
        layer_names.append(name)

    st.divider()
    
    # SECCIÓN DE INVERSIÓN Y FÍSICA
    st.subheader("⚙️ Motor WFRD")
    inv_type = st.radio("Dimensión", ["2D Isotrópica", "3D Anisótropa"])
    is_3d = "3D" in inv_type
    channel_suite = st.selectbox("Arreglo Real-Time", ["Full Suite", "Deep 100K", "Hi-Res 2MHz"])
    
    st.divider()
    
    # AJUSTE ESTRUCTURAL (EL CORAZÓN DEL MODELO)
    st.subheader("📐 Ajuste Estructural")
    current_dip = st.slider("DIP (°)", -25.0, 25.0, st.session_state.prev_dip)
    current_shift = st.slider("Shift Estructural (ft)", -50.0, 50.0, st.session_state.prev_shift)
    
    if st.button("📌 Fijar como Ghost (Referencia)"):
        st.session_state.prev_shift = current_shift
        st.session_state.prev_dip = current_dip
        st.rerun()

# --- 2. MOTOR DE CORTINA 2D/3D (VISUALIZACIÓN) ---
st.title(f"WFRD Proactive Suite v45 - {inv_type}")

# Paleta OWC Restaurada
OWC_HighRelief = [[0.0, "#000032"], [0.2, "#004696"], [0.4, "#64C8FF"], [0.5, "#786432"], [0.8, "#C89632"], [1.0, "#FFE696"]]

col_main, col_stats = st.columns([3, 1])

with col_main:
    # Generación de la Cortina con Capas Horizontales y Shift
    fig = go.Figure()
    
    # Simulación de MD y TVD Perp
    md = np.linspace(0, 1000, 100)
    tvd_p = np.linspace(-55, 55, 60)
    
    # Dibujo del GHOST PLOT (Sombra anterior)
    ghost_y = st.session_state.prev_shift + (md * np.tan(np.radians(st.session_state.prev_dip)))
    fig.add_trace(go.Scatter(x=md, y=ghost_y, name="Ghost (Anterior)", line=dict(color='rgba(255,165,0,0.4)', dash='dot')))

    # Dibujo de Capas Horizontales con Shift/DIP Actual
    # Aquí es donde se ve el "Corte de Capa"
    for i in range(int(n_layers)):
        y_pos = (i * 15 - 30) + current_shift + (md * np.tan(np.radians(current_dip)))
        fig.add_trace(go.Scatter(x=md, y=y_pos, name=layer_names[i], line=dict(width=2)))

    # Trayectoria del Pozo (Línea Blanca)
    fig.add_trace(go.Scatter(x=md, y=np.zeros_like(md), name="Trayectoria Pozo", line=dict(color='white', width=5)))

    fig.update_layout(height=600, template="plotly_dark", yaxis=dict(title="TVD Perpendicular (ft)", range=[55, -55]))
    st.plotly_chart(fig, use_container_width=True)

with col_stats:
    st.subheader("📊 Reporte WFRD")
    st.metric("DTBss Tope", f"{abs(current_shift - 15):.1f} ft")
    st.metric("DIP Aplicado", f"{current_dip}°")
    
    # El semáforo de caritas (v40)
    smile = "🙂" if abs(current_shift) < 12 else "🙁"
    st.markdown(f"<h1 style='text-align: center; font-size: 80px;'>{smile}</h1>", unsafe_allow_html=True)
    st.caption("Estatus de Navegación")

# --- 3. REPORTE DE GUARDIA ---
st.divider()
if st.button("📄 Generar Reporte de Guardia"):
    st.info(f"Reporte compilado para {n_layers} capas. Shift: {current_shift}ft. DIP: {current_dip}°.")
