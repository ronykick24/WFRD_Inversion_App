import numpy as np

def calculate_3d_horns(rh, rv, inc, dip, dist_to_bed):
    """Simula Cuernos de Polarización y Anisotropía."""
    alpha = np.radians(inc - dip)
    lam = np.sqrt(rv / (rh + 1e-6))
    # Efecto de borde: el cuerno crece al acercarse a la interfase
    horn_effect = 1.0 + (np.exp(-abs(dist_to_bed) / 3.0) * (lam - 1))
    return rh * horn_effect

def get_geo_metrics(md, inc, dip, shift, tst_target):
    """Calcula TVDss, TST, TVT y DTBss."""
    tvd = md * np.cos(np.radians(inc))
    tvdss = tvd - 5000  # Ajustar elevación según sea necesario
    # TVT (True Vertical Thickness)
    tvt = tst_target / np.cos(np.radians(dip))
    # DTBss (Distance to Bed - Subsea)
    dtbss_top = shift
    dtbss_base = shift + tst_target
    return tvdss, tvt, dtbss_top, dtbss_base
