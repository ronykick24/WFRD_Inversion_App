import numpy as np

def get_ahta_sensitivity(dist, freq='100kHz'):
    limit = {'100kHz': 18.4, '400kHz': 12.0, '2MHz': 6.0}.get(freq, 10.0)
    # Sensibilidad proactiva basada en ScienceDirect 2024
    if abs(dist) <= limit:
        return float(np.exp(-abs(dist) / (limit / 2.8)))
    return 0.0

def calculate_forward_model(rh, rv, inc, dip, dist):
    try:
        lam = np.sqrt(float(rv) / (float(rh) + 1e-6))
        rel_angle = np.radians(abs(float(inc) - float(dip)))
        sens = get_ahta_sensitivity(dist, '100kHz')
        # Respuesta azimutal según manual WFRD
        horn_effect = 1.0 + (sens * (lam - 1) * np.sin(rel_angle))
        return float(rh) * horn_effect
    except:
        return float(rh)

def calculate_tst(thickness, dip):
    """Calcula Total Stratigraphic Thickness."""
    return abs(float(thickness) * np.cos(np.radians(float(dip))))
