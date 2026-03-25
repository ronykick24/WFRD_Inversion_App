import numpy as np

def calculate_3d_horns(rh, rv, inc, dip, dist_to_bed):
    """Simula la física WFRD: Diferencia entre 2D (Isotrópico) y 3D (Anisotrópico)."""
    lam = np.sqrt(float(rv) / (float(rh) + 1e-6))
    # El límite físico de la herramienta: el efecto de cuerno es proactivo
    horn_effect = 1.0 + (np.exp(-abs(float(dist_to_bed)) / 3.5) * (lam - 1))
    return float(rh) * horn_effect

def get_geo_metrics(md, inc, dip, shift, tst_target):
    """Calcula el espesor verdadero (TST/TVT) y DTBss Estructural."""
    inc_rad, dip_rad = np.radians(float(inc)), np.radians(float(dip))
    tvd = float(md) * np.cos(inc_rad)
    tvdss = tvd - 5000 
    
    # TVT: Thickness normal a la trayectoria vs TST: Espesor real del estrato
    tvt = float(tst_target) / np.cos(dip_rad)
    # DTBss proyectado al límite de la capa (Top)
    dtbss_point = float(shift) + (float(md) * np.tan(dip_rad))
    
    return tvdss, tvt, dtbss_point
