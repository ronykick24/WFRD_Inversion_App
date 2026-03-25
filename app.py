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
    denom = np.sqrt(np.cos(theta_rel)**2 + lam_sq * np.sin(theta_res)**2 + 1e-12)
    return rh / denom

def forward_model_logic(m, md, inc, user_dip, n_layers):
    # m contiene: [resistividades (n), espesores (n-1), ani_ratio (1)]
    res_h = np.clip(m[:n_layers], 0.1, 1000)
    thick = np.clip(m[n_layers:2*n_layers-1], 1, 100)
    ani_ratio = np.clip(m[-1], 1.0, 5.0)
    
    alpha_rel = np.radians(inc - (90 + user_dip))
    tvd_p = md * np.sin(alpha_rel)
    z_int = np.cumsum(np.concatenate(([0], thick))) - np.sum(thick)/2
    
    # Perfil de capas
    rh_p = np.full_like(md, res_h[0], dtype=float)
    for i in range(len(z_int)-1):
        # Transición suave para estabilidad del optimizador
        w = 0.5 * (1 + np.tanh(np.clip((tvd_p - z_int[i])/3.0, -20, 20)))
        rh_p = rh_p * (1 - w) + res_h[i+1] * w
    
    ra = get_3d_anisotropy_ra(rh_p, rh_p * ani_ratio, inc, user_dip)
    
    # Cuernos de Polarización
    for zi in z_int:
        ra *= (1 + 1.2 * np.exp(-np.abs(tvd_p - zi) / 1.8))
    
    return np.nan_to_num(ra, nan=10.0) # Protege contra NaNs

# --- 2. CONFIGURACIÓN APP ---
st.set_page_config(layout="wide", page_title="WFRD Proactive v16")

with st.sidebar:
    st.title("🛠️ Configuración")
    calc_mode = st.selectbox("Algoritmo", ["Global (1000 iters)", "Local Fast"])
    res_ch = st.selectbox("Canal", ["AD2_GW6", "PD2_GW6", "AD4_GW6", "PU1_GW6"])
    dip_input = st.slider("DIP (°)", -20.0, 20.0, 0.0)
    n_layers = st.slider("Capas", 3, 7, 5)

uploaded_file = st.file_uploader("Cargar TSV", type=["tsv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep='\t')
    df.columns = [c.upper() for c in df.columns]
    
    # Limpieza estricta de datos
    clean_df = df[['MD', res_ch, 'INC']].apply(pd.to_numeric, errors='coerce').dropna()
    md = clean_df['MD'].values
    log_data = clean_df[res_ch].values
    last_inc = float(clean_df['INC'].iloc[-1])
    
    # --- INVERSIÓN ---
    # Límites: n capas res, n-1 espesores, 1 ani
    bounds = [(0.1, 1000)]*n_layers + [(2, 60)]*(n_layers-1) + [(1.0, 5.0)]
    
    def objective_func(m):
        pred = forward_model_logic(m, md, last_inc, dip_input, n_layers)
        # Asegura que tengan la misma longitud antes de restar
        if len(pred) != len(log_data): return 1e12
        # Error RMS en escala logarítmica con offset de seguridad
        err = np.sqrt(np.mean((np.log10(log_data + 1e-3) - np.log10(pred + 1e-3))**2))
        return err if np.isfinite(err) else 1e12

    with st.spinner("Invirtiendo física 3D..."):
        try:
            if "Global" in calc_mode:
                res_opt = differential_evolution(objective_func, bounds, maxiter=1000, popsize=10, polish=False).x
            else:
                x0 = [10]*n_layers + [15]*(n_layers-1) + [1.5]
                res_opt = least_squares(lambda m: np.log10(log_data+1e-3) - np.log10(forward_model_logic(m, md, last_inc, dip_input, n_layers)+1e-3), 
                                        x0=x0, bounds=([b[0] for b in bounds], [b[1] for b in bounds])).x
        except Exception as e:
            st.error(f"Falla en convergencia: {e}")
            st.stop()

    # Resultados
    res_h = res_opt[:n_layers]
    thick = res_opt[n_layers:2*n_layers-1]
    ani_final = res_opt[-1]
    interfaces = np.cumsum(np.concatenate(([0], thick))) - np.sum(thick)/2
    error_rms = objective_func(res_opt)

    # --- DASHBOARD ---
    st.markdown("### 🚦 Monitor de Inversión")
    st_c1, st_c2, st_c3 = st.columns(3)
    conf = "ALTA ✅" if error_rms < 0.07 else "MEDIA ⚠️" if error_rms < 0.15 else "BAJA ❌"
    st_c1.metric("Confianza", conf)
    st_c2.metric("Error RMS", f"{error_rms:.4f}")
    st_c3.metric("Anisotropía (Rv/Rh)", f"{ani_final:.2f}")

    # --- TRACKS Y MAPEO ---
    t_grid = np.linspace(-50, 50, 100)
    f_md = np.linspace(md[-1], md[-1] + 200, 50)
    md_tot = np.concatenate([md, f_md])
    dip_off = -(md_tot - md[0]) * np.tan(np.radians(dip_input))
    
    # Earth Model Sombreado
    z_map = np.zeros((len(t_grid), len(md_tot)))
    for j in range(len(md_tot)):
        idx = np.searchsorted(interfaces + dip_off[j], t_grid)
        z_map[:, j] = res_h[np.clip(idx, 0, n_layers-1)]

    fig = go.Figure(data=go.Heatmap(z=np.log10(z_map), x=md_tot, y=t_grid, colorscale="Turbo", opacity=0.9))
    
    # Trayectoria y DTBss
    well_y = np.concatenate([np.zeros(len(md)), (f_md-md[-1])*np.sin(np.radians(last_inc-90))])
    fig.add_trace(go.Scatter(x=md_tot, y=well_y, name="Pozo", line=dict(color='white', width=4)))
    
    dttb = abs(min([z for z in interfaces if z < 0], default=-50))
    dtbb = abs(max([z for z in interfaces if z > 0], default=50))
    fig.add_annotation(x=md[-1], y=dip_off[len(md)-1]-dttb, text=f"↑ TOP: {dttb:.1f}ft", bgcolor="lime")
    fig.add_annotation(x=md[-1], y=dip_off[len(md)-1]+dtbb, text=f"↓ BASE: {dtbb:.1f}ft", bgcolor="yellow")

    fig.update_layout(height=600, template="plotly_dark", yaxis=dict(title="TVD Perpendicular", autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

    # Tabla de Capas
    st.markdown("### 📊 Propiedades de Capas")
    st.table(pd.DataFrame({
        "Rh (Horiz)": [f"{r:.2f}" for r in res_h],
        "Rv (Vert)": [f"{r*ani_final:.2f}" for r in res_h],
        "DTBss (ft)": [f"{z:.2f}" for z in interfaces] + ["Extremo"]
    }))
