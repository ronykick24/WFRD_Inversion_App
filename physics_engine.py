import numpy as np

def calculate_3d_anisotropy(rh, rv, inc, dip, azim, strike):
    """
    Calcula la resistividad aparente (Ra) considerando inclinación, 
    dip, azimut del pozo y rumbo de la capa (Modelo 3D Relativo).
    """
    # Ángulo de ataque real en el espacio 3D
    # Simplificado a 2D-Plane para este motor, pero preparado para Azimut
    theta_res = np.radians(inc - dip) 
    
    # Lambda es la raíz de la relación de anisotropía
    lam = np.sqrt(rv / rh)
    
    # Ecuación de Ra para herramientas de propagación
    ra = rh / np.sqrt(np.cos(theta_res)**2 + (lam**2) * np.sin(theta_res)**2)
    return ra

def get_perpendicular_distance(md, inc, dip):
    """
    Calcula el TVD perpendicular (TVD_perp) a la formación. 
    Crucial para que el DTB sea geométricamente correcto.
    """
    alpha_rel = np.radians(inc - (90 + dip))
    return md * np.sin(alpha_rel)
