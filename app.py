import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import differential_evolution, least_squares

# --- SECCIÓN 1: MOTOR DE FÍSICA AVANZADA (INTERNAL) ---
def calculate_3d_anisotropy(rh, rv, inc, dip):
    theta_res = np.radians(inc - dip) 
    lam_sq = np.clip(rv / (rh + 1e-9), 1.0, 25.0)
    denom = np.sqrt(np.cos(theta_res)**2 + lam_sq * np.sin(theta_res)**2)
    return rh / (denom + 1e-12)

def apply_polarization_horns(res_array, md, interfaces, tvd_perp):
    """Simula cuernos de polarización en los contactos"""
    refined_res = res_array.copy()
    for z_int in interfaces:
        dist = np.abs(tvd_perp - z_int)
        horn_effect = 1.2 * np.exp(-dist / 2.5) # Efecto decaimiento exponencial
        refined_res *= (1 + horn_effect)
    return refined_res

def generate_azim_image(res_val, dttb, dtbb, n_bins=32):
    """Crea la respuesta 0-360 para visualización LWD"""
    angles = np.linspace(0, 2*np.pi, n_bins)
    # Simula que la señal es más fuerte hacia el límite más cercano
    sensitivity = np.cos(angles) * (1.0 / (dttb + 0.1) - 1.0 / (dtbb + 0.1))
    return res_val * (1 + 0.15 * sensitivity)

# --- SECCIÓN 2: MOTOR DE INVERSIÓN WFRD ---
class WFRD_Engine:
    def forward_model(self, m, md, inc, dip, n_layers):
        res_h = np.clip(m[:n_layers], 0.1, 1000)
        thick = np.clip(m[n_layers:2*n_layers-1], 1, 100)
        ani_ratio = m[-1]
        
        alpha_rel = np.radians(inc - (90 + dip))
        tvd_perp = md * np.sin(alpha_rel)
        z_int = np.cumsum(np.concatenate(([0], thick))) - np.sum(thick)/2
        
        rh_layer = np.full_like(md, res_h[0], dtype=float)
        for i in range(len(z_int)-1):
            weight = 0.5 * (1 + np.tanh((tvd_perp - z_int[i])/4.0))
            rh_layer = rh_layer * (1 - weight) + res_h[i+1] * weight
            
        ra = calculate_3d_anisotropy(rh_layer, rh_layer * ani_ratio, inc, dip)
        return apply_polarization_horns(ra, md, z_int, tvd_perp)

    def solve(self, mode, obs, md, inc, dip, n_layers):
        obs_np = np.nan_to_num(np.asarray(obs, dtype=float), nan=10.0)
        md_np = np.asarray(md, dtype=float)
        bounds = [(0.2, 1000)] * n_layers + [(2, 60)] * (n_layers - 1) + [(1.0, 5.0)]
        
        def obj(m):
            try:
                pred = self.forward_model(m, md_np, inc, dip, n_layers)
                return np.sqrt(np.mean((np.log10(obs_np) - np.log10(pred))**2))
            except: return 1e12

        if "Global" in mode:
            res = differential_evolution(obj, bounds=bounds, maxiter=1000, popsize=15, polish=False)
        else:
            x0 = [10]*n_layers + [15]*(n_layers-1) + [1.5]
            res = least_squares(lambda m: np.log10(obs_np) - np.log10(self.forward_model(m, md_np, inc, dip, n_layers)), 
                                x0=x0, bounds=([b[0] for b in bounds], [b[1] for b in bounds]), max_nfev=100)
        return res.x

# --- SECCIÓN 3: INTERFAZ DE USUARIO (STREAMLIT) ---
st.set_page_config(layout="wide", page_title="WFRD Proactive Steering")

st.sidebar.title("💎 Configuración LWD")
calc_mode = st.sidebar.selectbox("Estrategia", ["Estocástico Global (1000 iters)", "Determinístico Fast"])
res_ch = st.sidebar.selectbox("Canal Resistividad", ["AD2_GW6", "PD2_GW6", "AD4_GW6", "PU1_GW6"])
user_dip = st.sidebar.slider("DIP Formación (°)", -15.0, 15.0, 0.0)
n_layers = st.sidebar.slider("Número de Capas", 3, 9, 5)

uploaded_file = st.file_uploader("Cargar Datos (.tsv)", type=["tsv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep='\t')
    df.columns = [c.upper() for c in df.columns]
    md_array = pd.to_numeric(df['MD'], errors='coerce').dropna().values
    res_array = pd.to_numeric(df[res_ch], errors='coerce').loc[df['MD'].notna()].values
    last_inc = float(df['INC'].dropna().iloc[-1])
    
    engine = WFRD_Engine()
    with st.spinner('Procesando inversión multicapa y polarización...'):
        p = engine.solve(calc_mode, res_array, md_array, last_inc, user_dip, n_layers)

    res_h, thick, ani = p[:n_layers], p[n_layers:2*n_layers-1], p[-1]
    interfaces = np.cumsum(np.concatenate(([0], thick))) - np.sum(thick)/2

    # Cálculos de DTBss para la capa actual (zona de mayor resistividad)
    dttb = abs(min([z for z in interfaces if z < 0], default=-50))
    dtbb = abs(max([z for z in interfaces if z > 0], default=50))

    # --- LAYOUT PRINCIPAL ---
    c1, c2 = st.columns([3, 1])

    with c1:
        st.subheader("🗺️ Mapeo Geológico Invertido")
        t_grid = np.linspace(-60, 60, 100)
        f_md = np.linspace(md_array[-1], md_array[-1] + 150, 40)
        md_total = np.concatenate([md_array, f_md])
        dip_off = -(md_total - md_array[0]) * np.tan(np.radians(user_dip))
        
        z_map = np.zeros((len(t_grid), len(md_total)))
        for j in range(len(md_total)):
            idx = np.searchsorted(interfaces + dip_off[j], t_grid)
            z_map[:, j] = res_h[np.clip(idx, 0, n_layers-1)]

        fig = go.Figure(data=go.Heatmap(z=np.log10(z_map), x=md_total, y=t_grid, 
                                        colorscale="Turbo", colorbar=dict(title="Ra (log10)")))
        
        # Trayectoria y Labels DTBss
        well_y = np.concatenate([np.zeros(len(md_array)), (f_md-md_array[-1])*np.sin(np.radians(last_inc-90))])
        fig.add_trace(go.Scatter(x=md_total, y=well_y, name="Pozo", line=dict(color='white', width=3)))
        
        # Marcar los DTBss en el BIT
        fig.add_annotation(x=md_array[-1], y=dip_off[len(md_array)-1]-dttb, text=f"↑ DTB: {dttb:.1f}", font=dict(color="lime"))
        fig.add_annotation(x=md_array[-1], y=dip_off[len(md_array)-1]+dtbb, text=f"↓ DTB: {dtbb:.1f}", font=dict(color="yellow"))
        
        fig.update_layout(height=550, template="plotly_dark", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("🌀 Imagen Azimutal")
        azim_rows = [generate_azim_image(r, dttb, dtbb) for r in res_array[-30:]]
        fig_img = go.Figure(data=go.Heatmap(z=np.array(azim_rows), colorscale="YlOrBr", showscale=False))
        fig_img.update_layout(height=450, margin=dict(l=10, r=10, t=30, b=10), title="Visualización 0-360°")
        st.plotly_chart(fig_img, use_container_width=True)

    # --- TABLA DE RESULTADOS DE INVERSIÓN ---
    st.markdown("### 📊 Modelo de Capas y Límites de Formación")
    res_df = pd.DataFrame({
        "Capa": range(1, n_layers + 1),
        "Rh (Horizontal)": [f"{r:.2f}" for r in res_h],
        "Rv (Vertical)": [f"{r*ani:.2f}" for r in res_h],
        "DTBss Relativo (ft)": [f"{z:.2f}" for z in interfaces] + ["Infinito"]
    })
    st.table(res_df)
