import streamlit as st
import pandas as pd
import plotly.express as px
from utils import clean_wfrd_data
from engine_wfrd import InversionEngine

st.set_page_config(page_title="WFRD Inversion Pro", layout="wide")

st.title("🚀 WFRD GuideWave: Inversión Estocástica Multicapa")

uploaded_file = st.file_uploader("Cargar Hoja de cálculo (TSV)", type=["tsv"])

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file, sep='\t')
    df = clean_wfrd_data(df_raw)
    
    st.sidebar.header("Parámetros de Herramienta")
    sensor = st.sidebar.selectbox("Frecuencia/Espaciamiento", ['AD2_GW6', 'AD4_GW6', 'AU1_GW6'])
    
    # Ejecutar Inversión
    engine = InversionEngine()
    # Tomamos el valor actual para el cálculo
    current_val = df[sensor].iloc[0]
    inv_res = engine.run_stochastic_inversion(current_val)
    
    # Gráfico de Geonavegación (Paleta Azul -> Rojo)
    fig = px.scatter(df, x="MD", y=sensor, color=sensor,
                     color_continuous_scale=['#00008B', '#4169E1', '#DCDCDC', '#FF4500', '#8B0000'],
                     title="Corte de Resistividad de Formación (33-50 ft)")
    
    st.plotly_chart(fig, use_container_width=True)
    st.success(f"Inversión Completada: Resistividad Detectada {inv_res[0]:.2f} Ohm-m")
