import numpy as np

def calculate_3d_anisotropy(rh, rv, inc, dip):
    # Ángulo relativo al eje de la formación
    theta_res = np.radians(inc - dip) 
    # Relación de anisotropía Lambda^2
    lam_sq = np.clip(rv / (rh + 1e-9), 1.0, 25.0)
    # Respuesta aparente Ra
    denom = np.sqrt(np.cos(theta_res)**2 + lam_sq * np.sin(theta_res)**2)
    return rh / (denom + 1e-12)

def get_perpendicular_distance(md, inc, dip):
    # Distancia real perpendicular a la capa (considera el buzamiento)
    alpha_rel = np.radians(inc - (90 + dip))
    return md * np.sin(alpha_rel)
