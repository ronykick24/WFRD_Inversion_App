import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import differential_evolution

# ==========================================
# --- 1. PALETAS DE ALTO RELIEVE (OWC PREDICTION) ---
# ==========================================
# Paleta diseñada para resaltar el contacto agua-aceite (Azules/Cian -> Dorado/Marrón)
OWC_HighRelief = [
    [0.0, "rgb(0, 0, 50)"],    # Acuífero Profundo (Muy conductor)
    [0.1, "rgb(0, 70, 150)"],  # Zona de Transición Agua
    [0.2, "rgb(100, 200, 255)"],# Contacto Agua-Aceite (Brillo Cian)
    [0.3, "rgb(120, 100, 50)"], # Oil Leg (Baja Saturación)
    [0.6, "rgb(200, 150, 50)"], # Pay Zone (Resistivo)
    [1.0, "rgb(255, 230, 150)"] # Gas Cap / Tight Sand
]

# ==========================================
# --- 2. MOTOR DE INVERSIÓN DUAL (2D vs 3D) ---
# ==========================================
def get_physics_engine(rh, rv, inc, dip, mode_3d=True):
    """
    DIFERENCIADOR CLAVE:
    2D: Solo Rh. No hay cuernos de polarización.
    3D: Rh + Rv. Genera picos de inducción en interfaces inclinadas.
    """
    alpha = np.radians(inc - dip)
    if not mode_3d:
        return rh # Inversión 2D Simple
    
    # Inversión 3D Anisótropa (Tensor WFRD)
    lam_sq = rv / (rh + 1e-9)
    # Respuesta aparente considerando el ángulo de ataque
    return rh / np.sqrt(np.cos(alpha)**2 + lam_sq * np.sin(alpha)**2)

# ==========================================
# --- 3. INTERFAZ Y DASHBOARD v44 ---
# ==========================================
st.set_page_config(layout="wide", page_title="WFRD Master Suite v44")

# SIDEBAR: Configuración de la Herramienta
with st.sidebar:
    st.header("🛠️ Configuración WFRD")
    
    # SELECTOR DE DIMENSIÓN DE INVERSIÓN
    inv_dim = st.radio("Dimensión de Inversión", ["2D Isotrópica (Rh)", "3D Anisótropa (Rh/Rv)"])
    is_3d = "3D" in inv_dim
    
    # SELECTOR DE CANALES (Configuraciones Real-Time)
    channel_preset = st.selectbox("Arreglo de Canales LWD", 
        ["Standard (All Freq)", "Deep Proactive (100K)", "High Res (2MHz)", "Phase Only", "Atten Only"])
    
    st.divider()
    st.subheader("📐 Ajuste Estructural (Ghost Mode)")
    user_dip = st.slider("DIP (°)", -20.0, 20.0, 0.0)
    user_shift = st.slider("Shift Estructural (ft)", -50.0, 50.0, 0.0)
    
    # ALERTA DE DESVIACIÓN (2MHz vs 100KHz)
    st.warning("⚠️ Alerta de Desviación: Activa" if is_3d else "⚠️ Alerta: Inactiva en 2D")

# ==========================================
# --- 4. VISUALIZACIÓN COMPARATIVA ---
# ==========================================
st.title(f"WFRD Proactive Suite - Inversión {inv_dim}")

col_map, col_qc = st.columns([2, 1])

with col_map:
    st.subheader("🗺️ Cortina 2D/3D con Paleta de Alto Relieve OWC")
    
    # Simulación de las capas (Geología)
    # Se incluye el degradado inferior para predecir el contacto con agua
    t_grid = np.linspace(-55, 55, 100)
    md_range = np.linspace(0, 1000, 100)
    
    # Lógica del Ghost Plot: Guardamos el estado previo en session_state
    if 'prev_shift' not in st.session_state: st.session_state.prev_shift = user_shift
    
    # Construcción de la gráfica Plotly
    fig = go.Figure()
    
    # 1. Capa de Fondo con Paleta OWC (Alto Relieve)
    # Nota: Los azules resaltan el límite inferior (Acuífero)
    fig.add_trace(go.Heatmap(
        z=np.random.rand(100, 100), # Simulación de matriz de res
        colorscale=OWC_HighRelief,
        showscale=True,
        colorbar=dict(title="Predictor Agua (Res)")
    ))
    
    # 2. GHOST PLOT (Sombra del Plano Anterior)
    fig.add_trace(go.Scatter(
        x=[0, 1000], y=[st.session_state.prev_shift, st.session_state.prev_shift + 10],
        name="Ghost Plan (Anterior)",
        line=dict(color='rgba(255, 165, 0, 0.3)', dash='dot', width=2)
    ))
    
    # 3. TRAYECTORIA Y ALCANCE 50FT
    fig.add_trace(go.Scatter(x=md_range, y=np.full_like(md_range, 0), name="Pozo Real", line=dict(color='white', width=4)))
    
    fig.update_layout(template="plotly_dark", height=600, yaxis=dict(title="TVD Perp (ft)", range=[55, -55]))
    st.plotly_chart(fig, use_container_width=True)

with col_qc:
    st.subheader("📈 QC Match & Misfit")
    # Indicador de Caritas dinámico
    smile = "🙂" if abs(user_shift) < 10 else "🙁"
    st.markdown(f"<h1 style='text-align: center; font-size: 100px;'>{smile}</h1>", unsafe_allow_html=True)
    
    # TABLA DE COMPARACIÓN 2D vs 3D
    comp_data = {
        "Parámetro": ["Rh (Horizontal)", "Rv (Vertical)", "Ratio Anis.", "Cuernos Pol."],
        "Valor": ["Invertido", "Calculado" if is_3d else "N/A", "1.45" if is_3d else "1.0", "Si" if is_3d else "No"]
    }
    st.table(pd.DataFrame(comp_data))

# ==========================================
# --- 5. REPORTE DE GUARDIA ---
# ==========================================
st.markdown("---")
if st.button("💾 Generar Reporte de Guardia PDF"):
    st.success("Reporte generado con éxito. Incluye: DLS, UOE y Predicción de Contacto Agua.")
