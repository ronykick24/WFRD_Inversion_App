import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import differential_evolution, least_squares

# =========================================================
# SECCIÓN 1: FÍSICA Y ANISOTROPÍA 2D/3D (Original Protegida)
# =========================================================
def get_3d_anisotropy_ra(rh, rv, inc, dip):
    """
    Calcula la resistividad aparente (Ra) considerando el ángulo 
    relativo entre el pozo y el eje de anisotropía de la formación.
    """
    theta_rel = np.radians(inc - dip)
    # Lambda al cuadrado (Ratio de anisotropía)
    lam_sq = np.clip(rv / (rh + 1e-9), 1.0, 25.0)
    # Ecuación fundamental de Ra para herramientas de propagación
    denom = np.sqrt(np.cos(theta_rel)**2 + lam_sq * np.sin(theta_rel)**2)
    return rh / (denom + 1e-12)

def forward_model_logic(m, md, inc, user_dip, n_layers):
    """
    Simulación multicapa con efectos de borde y polarización.
    """
    res_h = np.clip(m[:n_layers], 0.1, 1000)
    thick = np.clip(m[n_layers:2*n_layers-1], 1, 100)
    ani_ratio = m[-1] # Rv/Rh
    
    # Geometría Perpendicular (DTBss)
    alpha_rel = np.radians(inc - (90 + user_dip))
    tvd_p = md * np.sin(alpha_rel)
    z_int = np.cumsum(np.concatenate(([0], thick))) - np.sum(thick)/2
    
    # Construcción del perfil de Rh
    rh_p = np.full_like(md, res_h[0], dtype=float)
    for i in range(len(z_int)-1):
        # Función de transición suave (Sigmoide) para evitar inestabilidad en la inversión
        w = 0.5 * (1 + np.tanh((tvd_p - z_int[i])/3.0))
        rh_p = rh_p * (1 - w) + res_h[i+1] * w
    
    # Cálculo 3D con Anisotropía
    rv_p = rh_p * ani_ratio
    ra = get_3d_anisotropy_ra(rh_p, rv_p, inc, user_dip)
    
    # SECCIÓN DE CUERNOS DE POLARIZACIÓN (Horns)
    # Aparecen en contactos de alta resistividad con alto ángulo
    for zi in z_int:
        dist = np.abs(tvd_p - zi)
        ra *= (1 + 1.2 * np.exp(-dist / 1.8))
    
    return ra

# =========================================================
# SECCIÓN 2: INTERFAZ Y DASHBOARD PROACTIVO
# =========================================================
st.set_page_config(layout="wide", page_title="Geo-Mapper 3D Proactive")

with st.sidebar:
    st.header("🛠️ Parámetros de Inversión")
    calc_mode = st.selectbox("Algoritmo", ["Global (1000 iteraciones)", "Local Determinístico"])
    res_ch = st.selectbox("Canal Log", ["AD2_GW6", "PD2_GW6", "AD4_GW6", "PU1_GW6"])
    dip_input = st.slider("DIP Formación (°)", -25.0, 25.0, 0.0)
    n_layers = st.slider("Capas en el Modelo", 3, 9, 5)

uploaded_file = st.file_uploader("Cargar Datos TSV", type=["tsv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep='\t')
    df.columns = [c.upper() for c in df.columns]
    
    # Limpieza de datos (Blindaje contra TypeErrors)
    md = pd.to_numeric(df['MD'], errors='coerce').dropna().values
    log_data = pd.to_numeric(df[res_ch], errors='coerce').loc[df['MD'].isin(md)].values
    last_inc = float(df['INC'].dropna().iloc[-1])
    
    # --- EJECUCIÓN DE LA INVERSIÓN ---
    bounds = [(0.1, 1000)]*n_layers + [(2, 60)]*(n_layers-1) + [(1.0, 5.0)]
    
    def objective_func(m):
        pred = forward_model_logic(m, md, last_inc, dip_input, n_layers)
        # Error cuadrático en escala LOG (Esencial para Geosteering)
        return np.sqrt(np.mean((np.log10(log_data + 1e-6) - np.log10(pred + 1e-6))**2))

    with st.spinner("🔄 Procesando Inversión 3D + Anisotropía..."):
        if "Global" in calc_mode:
            # Differential Evolution para evitar mínimos locales
            res_opt = differential_evolution(objective_func, bounds, maxiter=1000, popsize=15).x
        else:
            x0 = [10]*n_layers + [15]*(n_layers-1) + [1.5]
            res_opt = least_squares(lambda m: np.log10(log_data+1e-6) - np.log10(forward_model_logic(m, md, last_inc, dip_input, n_layers)+1e-6), 
                                    x0=x0, bounds=([b[0] for b in bounds], [b[1] for b in bounds])).x

    # Resultados Invertidos
    res_h = res_opt[:n_layers]
    thick = res_opt[n_layers:2*n_layers-1]
    ani_final = res_opt[-1]
    interfaces = np.cumsum(np.concatenate(([0], thick))) - np.sum(thick)/2
    error_rms = objective_func(res_opt)

    # --- INDICADOR DE CONFIANZA (SEMÁFORO) ---
    st.markdown("### 🚦 Monitor de Confianza de Inversión")
    st_c1, st_c2, st_c3, st_c4 = st.columns(4)
    
    conf_status = "ALTA ✅" if error_rms < 0.06 else "MEDIA ⚠️" if error_rms < 0.15 else "BAJA ❌"
    st_c1.metric("Confianza", conf_status)
    st_c2.metric("Error RMS", f"{error_rms:.4f}")
    st_c3.metric("Anisotropía (Rv/Rh)", f"{ani_final:.2f}")
    st_c4.metric("Capas Detectadas", n_layers)

    # --- TRACKS Y VALIDACIÓN GEOPÍSICA ---
    fig_val = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=("Validación Log vs Modelo", "Imagen Azimutal 0-360°"))
    fig_val.add_trace(go.Scatter(x=log_data, y=md, name="Log Real", line=dict(color='#00FFFF', width=2)), row=1, col=1)
    fig_val.add_trace(go.Scatter(x=forward_model_logic(res_opt, md, last_inc, dip_input, n_layers), y=md, 
                                 name="Inversión", line=dict(color='#FF3300', dash='dot')), row=1, col=1)
    
    # Imagen Azimutal Basada en Contraste
    dttb_temp = abs(min([z for z in interfaces if z < 0], default=-20))
    dtbb_temp = abs(max([z for z in interfaces if z > 0], default=20))
    azim_rows = []
    for r in log_data[-60:]:
        angles = np.linspace(0, 2*np.pi, 32)
        azim_rows.append(r * (1 + 0.2 * np.cos(angles) * (1/dttb_temp)))
    
    fig_val.add_trace(go.Heatmap(z=np.array(azim_rows), x=np.linspace(0,360,32), y=md[-60:], colorscale="YlOrBr", showscale=False), row=1, col=2)
    fig_val.update_yaxes(autorange="reversed")
    fig_val.update_xaxes(type="log", row=1, col=1)
    fig_val.update_layout(height=450, template="plotly_dark")
    st.plotly_chart(fig_val, use_container_width=True)

    # --- CORTINA DE GEOSTEERING CON DEGRADÉ (EARTH MODEL) ---
    st.subheader("🗺️ Mapeo Multicapa Proactivo (Earth Model)")
    t_grid = np.linspace(-60, 60, 150)
    f_md = np.linspace(md[-1], md[-1] + 300, 60) # Proyección a 300ft
    md_total = np.concatenate([md, f_md])
    dip_off = -(md_total - md[0]) * np.tan(np.radians(dip_input))
    
    # Sombreado de Modelo Tierra
    z_map = np.zeros((len(t_grid), len(md_total)))
    for j in range(len(md_total)):
        idx = np.searchsorted(interfaces + dip_off[j], t_grid)
        z_map[:, j] = res_h[np.clip(idx, 0, n_layers-1)]

    fig_map = go.Figure()
    # Heatmap con Barra de Colores Reintegrada
    fig_map.add_trace(go.Heatmap(z=np.log10(z_map), x=md_total, y=t_grid, 
                                 colorscale="Turbo", opacity=0.9, colorbar=dict(title="Ra Log10")))
    
    # Líneas de Boundary (DTBss Puntos)
    for zi in interfaces:
        fig_map.add_trace(go.Scatter(x=md_total, y=zi + dip_off, mode='lines', 
                                     line=dict(color='rgba(255,255,255,0.4)', width=1, dash='dash'), showlegend=False))

    # Trayectoria Wellbore
    well_y = np.concatenate([np.zeros(len(md)), (f_md-md[-1])*np.sin(np.radians(last_inc-90))])
    fig_map.add_trace(go.Scatter(x=md_total, y=well_y, name="Trayectoria", line=dict(color='black', width=4)))

    # Labels de Navegación Críticos (DTBss Arriba y Abajo)
    dttb = abs(min([z for z in interfaces if z < 0], default=-99))
    dtbb = abs(max([z for z in interfaces if z > 0], default=99))
    fig_map.add_annotation(x=md[-1], y=dip_off[len(md)-1]-dttb, text=f"↑ TOP: {dttb:.1f}ft", bgcolor="lime")
    fig_map.add_annotation(x=md[-1], y=dip_off[len(md)-1]+dtbb, text=f"↓ BASE: {dtbb:.1f}ft", bgcolor="yellow")

    fig_map.update_layout(height=650, template="plotly_dark", yaxis=dict(title="Distancia Perpendicular (ft)"))
    st.plotly_chart(fig_map, use_container_width=True)

    # --- TABLA DE RESULTADOS (EARTH MODEL DATA) ---
    st.markdown("### 📊 Propiedades de las Capas Invertidas")
    capas_df = pd.DataFrame({
        "Capa": range(1, n_layers+1),
        "Rh (Horiz)": [f"{r:.2f}" for r in res_h],
        "Rv (Vert)": [f"{r*ani_final:.2f}" for r in res_h],
        "DTBss (Distancia)": [f"{z:.2f} ft" for z in interfaces] + ["Extremo"]
    })
    st.dataframe(capas_df, use_container_width=True)
