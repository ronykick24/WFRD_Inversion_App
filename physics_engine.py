import numpy as np

def calculate_ahta_response(rh, rv, inc, dip, dist, freq='100kHz'):
    """
    Simula la respuesta de Fase y Atenuación del GuideWave.
    Implementa el límite físico de detección (DOD) del manual WFRD.
    """
    # Límites de detección del manual (DOD)
    dod_limits = {'100kHz': 18.5, '400kHz': 12.0, '2MHz': 6.0}
    limit = dod_limits.get(freq, 10.0)
    
    # Si la distancia supera el DOD, la herramienta solo ve la resistividad local (R0)
    if abs(dist) > limit:
        return float(rh)
    
    # Factor de Anisotropía Lambda (del manual)
    lam = np.sqrt(float(rv) / (float(rh) + 1e-6))
    rel_angle = np.abs(float(inc) - float(dip))
    
    # Simulación de la señal azimutal (Up/Down) para detectar fronteras
    # El efecto es más fuerte cuanto más cerca está la frontera y menor la frecuencia
    sensitivity = np.cos(np.radians(rel_angle)) * (limit - abs(dist)) / limit
    horn_effect = 1.0 + (sensitivity * (lam - 1))
    
    return float(rh) * horn_effect

def get_geo_metrics(md, inc, dip, shift):
    """Calcula TVDss y el DTBss Estructural Real."""
    tvdss = (float(md) * np.cos(np.radians(float(inc)))) - 5000
    # DTBss proyectado siguiendo la inclinación de la capa (Sección Estructural)
    dtbss_point = float(shift) + (float(md) * np.tan(np.radians(float(dip))))
    return tvdss, dtbss_point
