import numpy as np
from scipy.optimize import differential_evolution
from physics_engine import calculate_ahta_response

def run_ahta_inversion(res_vals, inc_vals, layers, progress_bar):
    """Inversión de 1000 iteraciones para resolver posición de capas."""
    res_vals = np.array(res_vals, dtype=float)
    inc_vals = np.array(inc_vals, dtype=float)
    mask = ~np.isnan(res_vals)
    r_c, i_c = res_vals[mask], inc_vals[mask]

    def objective(params):
        s_test, d_test = params
        # R0: Capa actual, R1: Techo, R2: Base
        rh, rv = float(layers[2]['rh']), float(layers[2]['rv'])
        
        errors = []
        for j in range(len(r_c)):
            # Modelo de propagación proactivo
            synth = calculate_ahta_response(rh, rv, i_c[j], d_test, s_test)
            errors.append((np.log10(r_c[j] + 1e-6) - np.log10(synth + 1e-6))**2)
        return np.mean(errors)

    progress_bar.progress(40, text="Calculando Inversión AHTA...")
    result = differential_evolution(objective, bounds=[(-60, 60), (-12, 12)], maxiter=1000)
    progress_bar.progress(100, text="Inversión WFRD Exitosa")
    return result.x
