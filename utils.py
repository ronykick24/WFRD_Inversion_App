import pandas as pd
import numpy as np

def clean_wfrd_data(df):
    # Eliminar la fila de unidades si existe
    if str(df['MD'].iloc[0]).startswith('.'):
        df = df.iloc[1:].copy()
    
    # Convertir todo a numérico, los textos se vuelven NaN
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Reemplazar nulos de Weatherford
    df = df.replace(-999.25, np.nan)
    
    # Eliminar filas donde MD sea nulo
    df = df.dropna(subset=['MD'])
    
    # Rellenar huecos pequeños para que el algoritmo no salte
    df = df.interpolate(method='linear', limit_area='inside')
    
    return df.reset_index(drop=True)
