import numpy as np

def calculate_3d_horns(rh, rv, inc, dip, dist_to_bed):
    """
    Simula la respuesta física de WFRD GW6.
    Calcula el efecto de anisotropía (3D) vs capas isotrópicas (2D).
    """
    try:
        rh, rv = float(rh), float(rv)
        # Factor de Anisotropía Lambda
        lam = np.sqrt(rv / (rh + 1e-6))
        # Ángulo relativo entre el pozo y la formación
        rel_angle = np.abs(float(inc) - float(dip))
        
        # Simulación de 'Horns' (Cuernos de polarización)
        # Aumentan exponencialmente al acercarse al límite físico (Boundary)
        scale_factor = np.exp(-np.abs(float(dist_to_bed)) / 3.8)
        horn_effect = 1.0 + (scale_factor * (lam - 1) * np.sin(np.radians(rel_angle)))
        
        return rh * horn_effect
    except:
        return rh

def get_geo_metrics(md, inc, dip, shift):
    """Cálculos de TVDss y DTBss Estructural (Punto de contacto)."""
    tvd = float(md) * np.cos(np.radians(float(inc)))
    tvdss = tvd - 5000 
    # El DTBss proyecta la posición del techo siguiendo el buzamiento (DIP)
    dtbss_point = float(shift) + (float(md) * np.tan(np.radians(float(dip))))
    return tvdss, dtbss_point
