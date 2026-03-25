import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import differential_evolution, least_squares

# --- 1. FÍSICA Y ANISOTROPÍA 2D/3D (Corrección de Variables) ---
def get_3d_anisotropy_ra(rh, rv, inc, dip):
    # Ángulo relativo entre el pozo y el buzamiento
    theta_rel = np.radians(inc - dip)
    # Coeficiente de anisotropía Lambda^2
    lam_sq = np.clip(rv / (rh + 1e-9), 1.0, 25.0)
    # Ecuación de Ra para herramientas de propagación electromagnética
    denom = np.sqrt(np.cos(theta_rel)**2 + lam_sq * np.sin(theta_rel)**2 + 1e-12)
    return rh / denom

def forward_model_logic(m, md, inc, user_dip, n_layers):
    # m: [res_h (n), thick (n-1), ani_ratio (1)]
    res_h = np.clip(m[:n_layers], 0.1, 1000)
    thick = np.clip(m[n_layers:2*n_layers-1], 1, 100)
    ani_ratio = np.clip(m[-1], 1.0, 5.0)
    
    # TVD Perpendicular relativo (DTBss)
    alpha_rel = np.radians(inc - (90 + user_dip))
    tvd_p = md * np.sin(alpha_rel)
    z_int = np.cumsum(np.concatenate(([0], thick))) - np.sum(thick)/2
    
    # Construcción del perfil de capas con transiciones suaves
    rh_p = np.full_like(md, res_h[0], dtype=float)
    for i in range(len(z_int)-1):
        # Sigmoide para suavizar los límites de capa y mejorar convergencia
        w = 0.5 * (1 + np.tanh(np.clip((tvd_p - z_int[i])/2.5, -20, 20)))
        rh_p = rh_p * (1 - w) + res_h[i+1] * w
    
    # Respuesta con Anisotropía 3D
    ra = get_3d_anisotropy_ra(rh_p, rh_p * ani_ratio, inc, user_dip)
    
    # Simulación de Cuernos de Polarización (Horns)
    for zi in z_int:
        dist = np.abs(tvd_p - zi)
        ra *= (1 + 1.2 * np.exp(-np.clip(dist / 1.5, 0, 50)))
    
    return np.nan_to_num(ra, nan=10.0)

# --- 2. CONFIGURACIÓN DE LA APP ---
st.set_page_config(layout="wide", page_title="WFRD Geo-Mapper v17")

with st.sidebar:
    st.header("
