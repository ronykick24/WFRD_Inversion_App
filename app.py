import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from engine_wfrd import WFRD_Engine_Core
from physics_engine import generate_azim_image

st.set_page_config(layout="wide", page_title="Advanced Geosteering Pro")

# --- SIDEBAR ---
st.sidebar.title("🛠️ Inversión y Física")
calc_mode = st.sidebar.selectbox("Modo", ["Estocástico Global (1000 iters)", "Determinístico"])
res_ch = st.sidebar.selectbox("Canal", ["AD2_GW6", "PD2_GW6", "AD4_GW6", "PU1_GW6"])
user_dip = st.sidebar.slider("DIP (°)", -15.0, 15.0, 0.0)
n_layers = st.sidebar.slider("Capas", 3, 9, 5)

uploaded_file = st.file_uploader("Cargar TSV", type=["tsv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep='\t')
    df.columns = [c.upper() for c in df.columns]
    md_array = pd.to_numeric(df['MD'], errors='coerce').dropna().values
    res_array = pd.to_numeric(df[res_ch], errors='coerce').loc[:len(md_array)-1].values
    last_inc = float(df['INC'].iloc[-1])
    
    engine = WFRD_Engine_Core()
    with st.spinner('Invirtiendo Modelo Multicapa...'):
        p, _ = engine.solve(calc_mode, res_array, md_array, last_inc, user_dip, n_layers)

    res_h, thick, ani = p[:n_layers], p[n_layers:2*n_layers-1], p[-1]
    interfaces = np.cumsum(np.concatenate(([0], thick))) - np.sum(thick)/2

    # --- CÁLCULO DE DTBss MULTICAPA ---
    # DTBss Arriba (positivos) y Abajo (negativos) respecto a la posición actual (0)
    dtb_list = interfaces # Distancias perpendiculares
    dttb = abs(min([z for z in interfaces if z < -0.1], default=-999))
    dtbb = abs(max([z for z in interfaces if z > 0.1], default=999))

    # --- VISUALIZACIÓN ---
    col1, col2 = st.columns([3, 1])

    with col1:
        # CORTINA CON BARRA DE COLORES (HEATMAP)
        tvd_grid = np.linspace(-60, 60, 100)
        f_md = np.linspace(md_array[-1], md_array[-1] + 150, 40)
        md_total = np.concatenate([md_array, f_md])
        dip_off = -(md_total - md_array[0]) * np.tan(np.radians(user_dip))
        
        z_map = np.zeros((len(tvd_grid), len(md_total)))
        for j in range(len(md_total)):
            idx = np.searchsorted(interfaces + dip_off[j], tvd_grid)
            z_map[:, j] = res_h[np.clip(idx, 0, n_layers-1)]

        fig = go.Figure(data=go.Heatmap(z=np.log10(z_map), x=md_total, y=tvd_grid, colorscale="Turbo", colorbar=dict(title="log10(Ra)")))
        fig.add_trace(go.Scatter(x=md_total, y=np.concatenate([np.zeros(len(md_array)), (f_md-md_array[-1])*np.sin(np.radians(last_inc-90))]), name="Pozo", line=dict(color='white')))
        
        # Labels DTBss
        fig.add_annotation(x=md_array[-1], y=dip_off[len(md_array)-1]-dttb, text=f"↑ DTB: {dttb:.1f}ft", font=dict(color="lime"))
        fig.add_annotation(x=md_array[-1], y=dip_off[len(md_array)-1]+dtbb, text=f"↓ DTB: {dtbb:.1f}ft", font=dict(color="orange"))
        
        fig.update_layout(height=500, template="plotly_dark", title="Sección de Mapeo Invertido")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # IMAGEN AZIMUTAL (LWD Style)
        st.write("🌀 Imagen Azimutal")
        azim_data = []
        for r_val in res_array[-20:]: # Últimos 20 samples
            azim_data.append(generate_azim_image(r_val, dttb, dtbb))
        
        fig_azim = go.Figure(data=go.Heatmap(z=np.array(azim_data), colorscale="YlOrBr"))
        fig_azim.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10), title="Resistividad Azimutal")
        st.plotly_chart(fig_azim, use_container_width=True)

    # --- TABLA DE LÍMITES (DTBss Multicapa) ---
    st.markdown("### 📊 Registro de Límites de Formación (Multicapa)")
    dtb_df = pd.DataFrame({
        "Interfase": [f"Límite {i+1}" for i in range(len(interfaces))],
        "DTBss Perpendicular (ft)": [f"{z:.2f}" for z in interfaces],
        "Tipo": ["ARRIBA (Techo)" if z < 0 else "ABAJO (Base)" for z in interfaces],
        "Resistividad Superior": [f"{res_h[i]:.1f}" for i in range(len(interfaces))],
        "Resistividad Inferior": [f"{res_h[i+1]:.1f}" for i in range(len(interfaces))]
    })
    st.dataframe(dtb_df, use_container_width=True)
