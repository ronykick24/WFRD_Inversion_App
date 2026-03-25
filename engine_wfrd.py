import numpy as np
from scipy.optimize import differential_evolution
from physics_engine import calculate_forward_model

def run_ahta_inversion(res_data, inc_data, layers, iterations=1000):
    res_vals = np.array(res_data, dtype=float)
    inc_vals = np.array(inc_data, dtype=float)
    mask = ~np.isnan(res_vals)
    r_c, i_c = res_vals[mask], inc_vals[mask]

    def objective(params):
        shift_t, dip_t = params
        rh, rv = layers[2]['rh'], layers[2]['rv'] 
        errors = [ (np.log10(r_c[j]+1e-6) - np.log10(calculate_forward_model(rh, rv, i_c[j], dip_t, shift_t + (j * np.tan(np.radians(dip_t))))+1e-6))**2 for j in range(len(r_c)) ]
        return np.mean(errors)

    res = differential_evolution(objective, bounds=[(-65.0, 65.0), (-15.0, 15.0)], maxiter=iterations)
    return [float(x) for x in res.x]
