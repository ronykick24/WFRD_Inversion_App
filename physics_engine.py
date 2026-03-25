import numpy as np

def get_ahta_sensitivity(dist, freq='100kHz'):
    """
    Define los límites de detección (DOD) según el Manual WFRD DS Azimuthal.
    100kHz es la frecuencia ultra-profunda para detección de fronteras.
    """
    dod_limits = {'100kHz': 18.4, '400kHz': 12.0, '2MHz': 6.0}
    limit = dod_limits.get(freq, 10.0)
    
    # Sensibilidad exponencial basada en ScienceDirect 2024
    if abs(dist) <= limit:
        return np.exp(-abs(dist) / (limit / 2.5))
    return 0

def calculate_forward_model(rh, rv, inc, dip, dist):
    """Modelo Forward Rápido para inversión proactiva."""
    try:
        # Anisotropía Lambda del manual WFRD
        lam = np.sqrt(float(rv) / (float(rh) + 1e-6))
        rel_angle = np.radians(abs(float(inc) - float(dip)))
        
        # Efecto de 'Horns' (Cuernos) cerca de la frontera estructural
        sensitivity = get_ahta_sensitivity(dist, '100kHz')
        # La respuesta azimutal depende del seno del ángulo relativo
        horn_effect = 1.0 + (sensitivity * (lam - 1) * np.sin(rel_angle))
        
        return float(rh) * horn_effect
    except:
        return float(rh)

def calculate_tst_tvt(measured_thickness, dip):
    """Calcula el espesor verdadero (TST) basado en el buzamiento."""
    dip_rad = np.radians(abs(dip))
    tst = measured_thickness * np.cos(dip_rad)
    return tst
