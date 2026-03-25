import numpy as np

def calculate_3d_anisotropy(rh, rv, inc, dip):
    theta_res = np.radians(inc - dip) 
    lam_sq = np.clip(rv / (rh + 1e-9), 1.0, 25.0)
    denom = np.sqrt(np.cos(theta_res)**2 + lam_sq * np.sin(theta_res)**2)
    return rh / (denom + 1e-12)

def apply_polarization_horns(res_array, md, interfaces, tvd_perp):
    """
    Simula los 'cuernos' de polarización en los límites de capa, 
    típicos en herramientas de propagación electromagnética.
    """
    refined_res = res_array.copy()
    for z_int in interfaces:
        # Los cuernos ocurren justo en el contacto (picos de resistividad aparente)
        dist_to_boundary = np.abs(tvd_perp - z_int)
        horn_effect = 1.5 * np.exp(-dist_to_boundary / 2.0) # Intensidad del pico
        refined_res *= (1 + horn_effect)
    return refined_res

def generate_azim_image(res_val, dttb, dtbb, n_bins=32):
    """
    Crea una imagen azimutal (0-360°) basada en la proximidad de los boundaries.
    """
    angles = np.linspace(0, 2*np.pi, n_bins)
    # El contraste aumenta si estamos cerca de un boundary (Up vs Down)
    up_down_ratio = np.clip(dttb / (dtbb + 1e-3), 0.5, 2.0)
    image_row = res_val * (1 + 0.2 * np.cos(angles) * (1/up_down_ratio))
    return image_row
import numpy as np

def calculate_3d_horns(rh, rv, inc, dip, dist_to_bed):
    """Genera los Cuernos de Polarización de Weatherford."""
    alpha = np.radians(inc - dip)
    lam = np.sqrt(rv / (rh + 1e-6))
    # El efecto aumenta cuando estás cerca de la interfase (< 5 ft)
    horn = 1.0 + (np.exp(-abs(dist_to_bed)/2.0) * (lam - 1))
    return rh * horn

def get_geo_metrics(md, inc, dip, shift, total_tst):
    """Cálculo de TVDss, TST, TVT y DTBss."""
    tvd = md * np.cos(np.radians(inc))
    tvdss = tvd - 5000  # Ajustar según tu elevación real
    tst = total_tst
    tvt = tst / np.cos(np.radians(dip))
    # DTBss (Distancia al tope/base ajustada por el Shift)
    dtbss_top = shift 
    dtbss_base = shift + tst
    return tvdss, tst, tvt, dtbss_top, dtbss_base
