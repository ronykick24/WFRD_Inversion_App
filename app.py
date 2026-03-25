import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import differential_evolution, least_squares

# --- 1. MOTOR DE FÍSICA Y RESPUESTA (Forward Model) ---
def forward_engine(m, md, inc, user_dip, n_layers):
    res_h = np.clip(m[:n_layers], 0.1, 1000)
    thick = np.clip(m[n_layers:2*n_layers-1], 1, 100)
    ani = m[-1] # Relación Rv/Rh
    
    # Geometría Perpendicular
    alpha_rel = np.radians(inc - (90 + user_dip))
    tvd_p = md * np.sin(alpha_rel)
    z_int = np.cumsum(np.concatenate(([0], thick))) - np.sum(thick)/2
    
    # Perfil de capas (Interpolación suave para evitar saltos numéricos)
    rh_profile = np.full_like(md, res_h[0], dtype=float)
    for i in range(len(z_int)-1):
        w = 0.5 * (1 + np.tanh((tvd_p - z_int[i])/3.0))
        rh_profile = rh_profile * (1 - w) + res_h[i+1] * w
    
    # Respuesta Aparente con Anisotropía
    theta = np.radians(inc - user_dip)
    ra = rh_profile / np.sqrt(np.cos(theta)**2 + (ani**2) * np.sin(theta)**2 + 1e-10)
    
    # Cuernos de Polarización en contactos
    for zi in z_int:
        ra *= (1 + 1.2 * np.exp(-np.abs(tvd_p - zi) / 1.5))
    return ra

# --- 2. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(layout="wide", page_title="WFRD Proactive Steering v15")

with st.sidebar:
    st.title("💎 Configuración LWD")
    mode = st.selectbox("Algoritmo de Inversión", ["Global (1000 iters)", "Local Fast"])
    res_ch = st.selectbox("Canal Resistividad", ["AD2_GW6", "PD2_GW6", "AD4_GW6", "PU1_GW6"])
    user_dip = st.slider("DIP Formación (°)", -20.0, 20.0, 0.0)
    n_layers = st.slider("Número de Capas", 3, 9, 5)
    st.info("El indicador de confianza evalúa el ajuste entre el modelo y el log real.")

# --- 3. CARGA Y PROCESAMIENTO ---
uploaded_file = st.file_uploader("Cargar Registro (.tsv)", type=["tsv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep='\t')
    df.columns = [c.upper() for c in df.columns]
    
    md = pd.to_numeric(df['MD'], errors='coerce').dropna().values
    log_res = pd.to_numeric(df[res_ch], errors='coerce').loc[df['MD'].isin(md)].values
    inc_val = float(df['INC'].dropna().iloc[-1])
    
    # --- PROCESO DE INVERSIÓN ---
    bounds = [(0.1, 1000)]*n_layers + [(2, 60)]*(n_layers-1) + [(1, 5)]
    
    def objective(m):
        pred = forward_engine(m, md, inc_val, user_dip, n_layers)
        # Error en espacio logarítmico para mejor resolución en bajas resistividades
        return np.sqrt(np.mean((np.log10(log_res + 1e-5) - np.log10(pred + 1e-5))**2))

    with st.spinner("🤖 Invirtiendo Modelo Tierra..."):
        if "Global" in mode:
            res_opt = differential_evolution(objective, bounds, maxiter=1000, popsize=15).x
        else:
            x0 = [10]*n_layers + [15]*(n_layers-1) + [1.5]
            res_opt = least_squares(lambda m: np.log10(log_res+1e-5) - np.log10(forward_engine(m, md, inc_val, user_dip, n_layers)+1e-5), 
                                    x0=x0, bounds=([b[0] for b in bounds], [b[1] for b in bounds])).x

    # Resultados post-inversión
    res_h = res_opt[:n_layers]
    thick = res_opt[n_layers:2*n_layers-1]
    ani_val = res_opt[-1]
    interfaces = np.cumsum(np.concatenate(([0], thick))) - np.sum(thick)/2
    final_error = objective(res_opt)

    # --- INDICADOR DE CONFIANZA (SEMÁFORO) ---
    st.markdown("### 🚦 Estado de la Inversión")
    c1, c2, c3, c4 = st.columns(4)
    
    # Lógica del semáforo
    if final_error < 0.05:
        conf_label, conf_col = "ALTA (Ajuste Perfecto)", "#00FF00"
    elif final_error < 0.15:
        conf_label, conf_col = "MEDIA (Revisar DIP)", "#FFFF00"
    else:
        conf_label, conf_col = "BAJA (Modelo no converge)", "#FF0000"

    c1.metric("Confianza", conf_label)
    c2.metric("Error RMS", f"{final_error:.4f}")
    c3.metric("Anisotropía (Rv/Rh)", f"{ani_val:.2f}")
    c4.metric("DIP Aplicado", f"{user_dip}°")

    # --- TRACKS Y VALIDACIÓN ---
    fig_val = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=("Curva vs Modelo", "Imagen Azimutal 0-360°"))
    fig_val.add_trace(go.Scatter(x=log_res, y=md, name="Real", line=dict(color='cyan', width=2)), row=1, col=1)
    fig_val.add_trace(go.Scatter(x=forward_engine(res_opt, md, inc_val, user_dip, n_layers), y=md, 
                                 name="Invertido", line=dict(color='red', dash='dot')), row=1, col=1)
    
    # Imagen Azimutal Sintética
    azim_img = np.array([np.full(32, r) * (1 + 0.2*np.cos(np.linspace(0,2*np.pi,32))) for r in log_res[-60:]])
    fig_val.add_trace(go.Heatmap(z=azim_img, x=np.linspace(0, 360, 32), y=md[-60:], colorscale="YlOrBr", showscale=False), row=1, col=2)
    
    fig_val.update_yaxes(autorange="reversed")
    fig_val.update_xaxes(type="log", row=1, col=1)
    fig_val.update_layout(height=450, template="plotly_dark")
    st.plotly_chart(fig_val, use_container_width=True)

    # --- CORTINA DE GEOSTEERING CON DEGRADÉ ---
    st.subheader("🗺️ Mapeo Multicapa y DTBss (Earth Model)")
    t_grid = np.linspace(-60, 60, 120)
    f_md = np.linspace(md[-1], md[-1] + 250, 50)
    md_total = np.concatenate([md, f_md])
    dip_off = -(md_total - md[0]) * np.tan(np.radians(user_dip))
    
    # Modelo de tierra con degradé
    z_map = np.zeros((len(t_grid), len(md_total)))
    for j in range(len(md_total)):
        idx = np.searchsorted(interfaces + dip_off[j], t_grid)
        z_map[:, j] = res_h[np.clip(idx, 0, n_layers-1)]

    fig_map = go.Figure()
    fig_map.add_trace(go.Heatmap(z=np.log10(z_map), x=md_total, y=t_grid, 
                                 colorscale="Turbo", opacity=0.85, colorbar=dict(title="Res (log)")))
    
    # Dibujar líneas de DTBss (Boundaries)
    for i, zi in enumerate(interfaces):
        fig_map.add_trace(go.Scatter(x=md_total, y=zi + dip_off, mode='lines', 
                                     line=dict(color='rgba(255,255,255,0.3)', width=1, dash='dash'), showlegend=False))

    # Trayectoria Proyectada
    well_path = np.concatenate([np.zeros(len(md)), (f_md-md[-1])*np.sin(np.radians(inc_val-90))])
    fig_map.add_trace(go.Scatter(x=md_total, y=well_path, name="Well", line=dict(color='black', width=4)))

    # Labels de Navegación en el BIT
    dttb = abs(min([z for z in interfaces if z < 0], default=-99))
    dtbb = abs(max([z for z in interfaces if z > 0], default=99))
    
    fig_map.add_annotation(x=md[-1], y=dip_off[len(md)-1]-dttb, text=f"↑ TOP: {dttb:.1f}ft", bgcolor="lime", font=dict(color="black"))
    fig_map.add_annotation(x=md[-1], y=dip_off[len(md)-1]+dtbb, text=f"↓ BASE: {dtbb:.1f}ft", bgcolor="yellow", font=dict(color="black"))

    fig_map.update_layout(height=650, template="plotly_dark", yaxis=dict(title="TVD Perpendicular"))
    st.plotly_chart(fig_map, use_container_width=True)

    # --- TABLA DE RESULTADOS FINALES ---
    st.markdown("### 📊 Propiedades del Modelo Tierra")
    res_df = pd.DataFrame({
        "Capa": range(1, n_layers + 1),
        "Rh (ohm.m)": [f"{r:.2f}" for r in res_h],
        "Rv (ohm.m)": [f"{r*ani_val:.2f}" for r in res_h],
        "DTBss Perpendicular (ft)": [f"{z:.2f}" for z in interfaces] + ["Infinito"]
    })
    st.dataframe(res_df, use_container_width=True)
