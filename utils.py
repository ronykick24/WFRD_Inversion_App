import pandas as pd
import numpy as np

def clean_wfrd_data(df):
    # 1. Eliminar la fila de unidades (fila 0 que contiene '.FT', '.deg', '.dB')
    if str(df['MD'].iloc[0]).startswith('.'):
        df = df.iloc[1:].copy()
    
    # 2. Reemplazar el nulo estándar de Weatherford (ya sea que se haya leído como string o número)
    df = df.replace(['-999.25', -999.25, '-999.250'], np.nan)
    
    # 3. Forzar conversión a numérico. 
    # errors='coerce' transformará textos como "Cerca del TECHO" en NaN (nulos matemáticos)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 4. Limpieza: Descartar filas sin datos en los sensores críticos
    critical_cols = ['AD2_GW6', 'PD2_GW6', 'AD4_GW6', 'AU1_GW6']
    valid_cols = [c for c in critical_cols if c in df.columns]
    
    if valid_cols:
        df = df.dropna(subset=valid_cols, how='all')
    
    # Asegurarnos de que el MD no tenga nulos para que el gráfico funcione
    df = df.dropna(subset=['MD'])
    
    return df.reset_index(drop=True)
