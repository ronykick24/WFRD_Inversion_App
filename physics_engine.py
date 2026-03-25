import numpy as np

def calculate_3d_horns(rh, rv, inc, dip, dist_to_bed):
    """Simula respuesta 3D (Anisotropía) vs 2D."""
    # Evitar división por cero y asegurar tipos float
    rh, rv, inc, dip, dist_to_bed = map(float, [rh, rv, inc, dip, dist_to_bed])
    lam = np.sqrt(rv / (rh + 1e-6))
    # Efecto de cuerno exponencial para simular aproximación al límite
    horn_effect = 1.0 + (np.exp(-abs(dist_to_bed) / 3.5) * (lam - 1))
    return rh * horn_effect

def get_geo_metrics(md, inc, dip, shift):
    """Cálculos de TVDss y Proyección Estructural del límite."""
    tvd = float(md) * np.cos(np.radians(float(inc)))
    tvdss = tvd - 5000 
    # El DTBss sigue la estructura de la capa (Estructural)
    dtbss_point = float(shift) + (float(md) * np.tan(np.radians(float(dip))))
    return tvdss, dtbss_point
