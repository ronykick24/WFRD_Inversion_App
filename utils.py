import pandas as pd
import numpy as np

def clean_wfrd_data(df):
    # Reemplazar los nulos de Weatherford (-999.25) por NaN de Python
    df = df.replace(-999.25, np.nan)
    # Eliminar filas donde las herramientas principales no tengan señal
    df = df.dropna(subset=['AD2_GW6', 'PD2_GW6'], how='all')
    return df

def calculate_apparent_res(ad_signal):
    # Función simplificada para convertir atenuación en Resistividad Aparente
    return 10**(ad_signal / 20)
