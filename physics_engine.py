import numpy as np

def calculate_3d_horns(rh, rv, inc, dip, dist_to_bed):
    """
    Simula la respuesta física 3D de Weatherford.
    Genera los picos (horns) cuando el pozo se acerca a un límite de capa.
    """
    alpha = np.radians(inc - dip)
    # Anisotropía (Lambda)
    lam = np.sqrt(rv / (rh + 1e-6))
    
    # Física de proximidad: El efecto aumenta exponencialmente al acercarse (< 5 ft)
    horn_effect = 1.0 + (np.exp(-abs(dist_to_bed) / 3.0) * (lam - 1))
    return rh * horn_effect

def get_geo_metrics(md, inc, dip, shift, tst_target):
    """Calcula parámetros de ingeniería: TVDss, TVT y DTBss."""
    tvd = md * np.cos(np.radians(inc))
    tvdss = tvd - 5000  # Elevación de referencia (KB)
    
    # Espesor Verdadero Vertical (TVT)
    tvt = tst_target / np.cos(np.radians(dip))
    
    # Distancia al lecho (DTBss)
    dtbss_top = shift
    return tvdss, tvt, dtbss_top
