import numpy as np

def calculate_3d_horns(rh, rv, inc, dip, dist_to_bed):
    """Simula Cuernos de Polarización y Anisotropía GW6."""
    alpha = np.radians(inc - dip)
    lam = np.sqrt(rv / (rh + 1e-6))
    horn_effect = 1.0 + (np.exp(-abs(dist_to_bed) / 3.0) * (lam - 1))
    return rh * horn_effect

def get_geo_metrics(md, inc, dip, shift, tst_target):
    """Calcula TVDss, TVT y DTBss."""
    tvd = md * np.cos(np.radians(inc))
    tvdss = tvd - 5000  # Ajustar elevación según sea necesario
    tvt = tst_target / np.cos(np.radians(dip))
    dtbss_top = shift
    return tvdss, tvt, dtbss_top
