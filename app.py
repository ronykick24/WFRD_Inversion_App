import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import differential_evolution, least_squares

# --- 1. FÍSICA Y ANISOTROPÍA 2D/3D ---
def get_3d_anisotropy_ra(rh, rv, inc, dip):
    theta_rel = np.radians(inc - dip)
    lam_sq = np.clip(rv / (rh + 1e-9), 1.0, 25.0)
    denom = np.sqrt(np.cos(theta_rel)**2 + lam_sq * np.sin(theta_rel)**2 + 1e-12)
    return rh / denom

def forward_model_logic(m, md, inc, user_dip, n_layers):
    res_h = np.clip(m[:n_layers], 0.1, 1000)
    thick = np.clip(m[n_layers:2*n_layers-1], 1, 100)
    ani_ratio = np.clip(m[-1], 1.0, 5.0)
    
    alpha_rel = np.radians(inc - (90 + user_dip))
    tvd_p = md * np.sin(alpha_rel)
    z_int = np.cumsum(np.concatenate(([0], thick))) - np.sum(thick)/2
    
    rh_p = np.full_like(md, res_h[0], dtype=float)
    for i in range(len(z_int)-1):
        w = 0.5 * (1 + np.tanh(np.clip((tvd_p - z_int[i])/2.5, -20, 20)))
        rh_p = rh_p * (1 - w) + res_h[i+1] * w
    
    ra = get_3d_anisotropy_ra(rh_p, rh_p * ani_ratio, inc, user_dip)
    for zi in z_int:
        dist = np.abs(tvd_p - zi)
        ra *= (1 + 1.2 * np.exp(-np.clip(dist / 1.5, 0, 50)))
    return np.nan_to_num(ra, nan=10.0)

# --- 2. CONFIGURACIÓN DE LA APP ---
st.set_page_config(layout="wide", page_title="WFRD Proactive Steering")

with st.sidebar:
    st.header("Configuración LWD")
    calc_mode = st.selectbox("Algoritmo", ["Global (1000 iters)", "Local Fast"])
    res_ch = st.selectbox("Canal Resistividad", ["AD2_GW6", "PD2_GW6", "AD4_GW6", "PU1_GW6"])
    dip_input = st.slider("DIP Formación (°)", -25.0, 25.0, 0.0)
    n_layers = st.slider("Capas en el Modelo", 3, 7, 5)

uploaded_file = st.file_uploader("Cargar Registro (.tsv)", type=["tsv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep='\t')
    df.columns = [c.upper() for c in df.columns]
    
    clean_df = df[['MD', res_ch, 'INC']].apply(pd.to_numeric, errors='coerce').dropna()
    md_vals = clean_df['MD'].values
    log_vals = clean_df[res_ch].values
    last_inc = float(clean_df['INC'].iloc[-1])
    
    bounds = [(0.1, 1000)]*n_layers + [(2, 60)]*(n_layers-1) + [(1.0, 5.0)]
    
    def objective_func(m):
        pred = forward_model_logic(m, md_vals, last_inc, dip_input, n_layers)
        err = np.sqrt(np.mean((np.log10(log_vals + 1e-3) - np.log10(pred + 1e-3))**2))
        return err if np.isfinite(err) else 1e12

    with st.spinner("Invirtiendo física 3D..."):
        try:
            if "Global" in calc_mode:
                res_opt = differential_evolution(objective_func, bounds, maxiter=1000, popsize=10, polish=False).x
            else:
                x0 = [10]*n_layers + [15]*(n_layers-1) + [1.5]
                res_opt = least_squares(lambda m: np.log10(log_vals+1e-3) - np.log10(forward_model_logic(m, md_vals, last_inc, dip_input, n_layers)+1e-3), 
                                        x0=x0, bounds=([b[0] for b in bounds], [b[1] for b in bounds])).x
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    res_h, thick, ani_val = res_opt[:n_layers], res_opt[n_layers:2*n_layers-1], res_opt[-1]
    interfaces = np
