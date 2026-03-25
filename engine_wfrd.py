import numpy as np
from scipy.optimize import differential_evolution
from physics_engine import calculate_3d_horns

def run_proactive_inversion(res_vals, inc_vals, layers):
    # Limpiamos posibles NaNs antes de procesar
    mask = ~np.isnan(res_vals) & ~np.isnan(inc_vals)
    r_clean = res_vals[mask]
    i_clean = inc_vals[mask]

    def objective(params):
        s_test, d_test = params
        rh, rv = layers[2]['rh'], layers[2]['rv']
        
        errors = []
        for j in range(len(r_clean)):
            synth = calculate_3d_horns(rh, rv, i_clean[j], d_test, s_test)
            errors.append((np.log10(r_clean[j] + 1e-6) - np.log10(synth + 1e-6))**2)
        return np.mean(errors)

    result = differential_evolution(objective, bounds=[(-60, 60), (-15, 15)])
    return result.x
