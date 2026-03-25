import numpy as np

def calculate_3d_horns(rh, rv, inc, dip, dist_to_bed):
    """Simula respuesta 3D (Anisotropía) vs 2D (Isotrópico)."""
    lam = np.sqrt(rv / (rh + 1e-6))
    # Efecto 3D: Cuernos de polarización por anisotropía
    horn_effect = 1.0 + (np.exp(-abs(dist_to_bed) / 3.5) * (lam - 1))
    return rh * horn_effect

def get_geo_metrics(md, inc, dip, shift, tst_target):
    """Cálculos de TVDss y Proyección Estructural."""
    tvd = md * np.cos(np.radians(inc))
    tvdss = tvd - 5000 
    # DTBss que sigue la capa
    dtbss_point = shift + (md * np.tan(np.radians(dip)))
    return tvdss, dtbss_point
