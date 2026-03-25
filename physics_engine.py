import numpy as np

def calculate_3d_anisotropy(rh, rv, inc, dip):
    # Ángulo respecto al eje de la formación (Eje de Anisotropía)
    theta_res = np.radians(inc - dip) 
    # Lambda (Anisotropía)
    lam_sq = np.clip(rv / (rh + 1e-9), 1.0, 25.0)
    # Ecuación de Ra: Evitamos división por cero
    denom = np.sqrt(np.cos(theta_res)**2 + lam_sq * np.sin(theta_res)**2)
    return rh / (denom + 1e-12)

def get_perpendicular_distance(md, inc, dip):
    # Distancia perpendicular real a la formación (90 + dip)
    alpha_rel = np.radians(inc - (90 + dip))
    return md * np.sin(alpha_rel)
