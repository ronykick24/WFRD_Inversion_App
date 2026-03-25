import numpy as np
from scipy.optimize import differential_evolution
from physics_engine import calculate_3d_horns

def run_proactive_inversion(res_vals, inc_vals, layers, progress_bar):
    """Inversión estocástica de alta intensidad (hasta 1000 iteraciones)."""
    # Forzar conversión a float y limpiar NaNs
    res_vals = np.array(res_vals, dtype=float)
    inc_vals = np.array(inc_vals, dtype=float)
    
    mask = ~np.isnan(res_vals)
    r_c, i_c = res_vals[mask], inc_vals[mask]

    def objective(params):
        s_test, d_test = params
        rh, rv = float(layers[2]['rh']), float(layers[2]['rv'])
        errors = []
        for j in range(len(r_c)):
            synth = calculate_3d_horns(rh, rv, i_c[j], d_test, s_test)
            errors.append((np.log10(r_c[j] + 1e-6) - np.log10(synth + 1e-6))**2)
        return np.mean(errors)

    # Inversión determinística de 1000 iteraciones para multicapa
    progress_bar.progress(50, text="Iterando modelos estocásticos...")
    result = differential_evolution(objective, bounds=[(-70, 70), (-15, 15)], tol=0.01)
    progress_bar.progress(100, text="Inversión Completada")
    return result.x # Retorna [Best_Shift, Best_Dip]
