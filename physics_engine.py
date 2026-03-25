import numpy as np

def get_ahta_sensitivity(dist, freq='100kHz'):
    """Límites de detección (DOD) según manual WFRD (18.4 ft)."""
    dod_limits = {'100kHz': 18.4, '400kHz': 12.0, '2MHz': 6.0}
    limit = dod_limits.get(freq, 10.0)
    return np.exp(-abs(dist) / (limit / 2.8)) if abs(dist) <= limit else 0.0

def calculate_forward_model(rh, rv, inc, dip, dist):
    """Modelo Forward: Respuesta de resistividad azimutal."""
    try:
        lam = np.sqrt(float(rv) / (float(rh) + 1e-6))
        rel_angle = np.radians(abs(float(inc) - float(dip)))
        sens = get_ahta_sensitivity(dist, '100kHz')
        horn_effect = 1.0 + (sens * (lam - 1) * np.sin(rel_angle))
        return float(rh) * horn_effect
    except:
        return float(rh)

def calculate_tst(thickness, dip):
    """Calcula el espesor estratigráfico real (TST)."""
    return abs(float(thickness) * np.cos(np.radians(float(dip))))
