import numpy as np

def get_ahta_sensitivity(dist, freq='100kHz'):
    """Límites de Detección (DOD) según Manual WFRD DS Azimuthal (18.4 ft)."""
    dod_limits = {'100kHz': 18.4, '400kHz': 12.0, '2MHz': 6.0}
    limit = dod_limits.get(freq, 10.0)
    if abs(dist) <= limit:
        return np.exp(-abs(dist) / (limit / 2.5))
    return 0

def calculate_forward_model(rh, rv, inc, dip, dist):
    """Modelo Forward de respuesta azimutal."""
    try:
        lam = np.sqrt(float(rv) / (float(rh) + 1e-6))
        rel_angle = np.radians(abs(float(inc) - float(dip)))
        sensitivity = get_ahta_sensitivity(dist, '100kHz')
        # Efecto de polarización (Horns) basado en ScienceDirect 2024
        horn_effect = 1.0 + (sensitivity * (lam - 1) * np.sin(rel_angle))
        return float(rh) * horn_effect
    except:
        return float(rh)

def calculate_tst(thickness, dip):
    """Calcula Total Stratigraphic Thickness (TST) corregido por DIP."""
    return thickness * np.cos(np.radians(abs(dip)))
