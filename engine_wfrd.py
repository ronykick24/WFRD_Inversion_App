import numpy as np
from scipy.optimize import differential_evolution
from physics_engine import calculate_3d_horns

def run_proactive_inversion(res_vals, inc_vals, layers, iterations=1000):
    """Inversión estocástica determinística para multicapa."""
    # Limpieza de datos
    res_vals = np.array(res_vals, dtype=float)
    inc_vals = np.array(inc_vals, dtype=float)
    mask = ~np.isnan(res_vals)
    r_c, i_c = res_vals[mask], inc_vals[mask]

    def objective(params):
        s_test, d_test = params
        # Enfocado en el Target Reservoir (Capa index 2)
        rh, rv = float(layers[2]['rh']), float(layers[2]['rv'])
        
        errors = []
        for j in range(len(r_c)):
            # Distancia al techo según el modelo propuesto
            dist = s_test + (j * np.tan(np.radians(d_test)))
            synth = calculate_3d_horns(rh, rv, i_c[j], d_test, dist)
            errors.append((np.log10(r_c[j] + 1e-6) - np.log10(synth + 1e-6))**2)
        return np.mean(errors)

    # Motor estocástico de alta intensidad
    result = differential_evolution(
        objective, 
        bounds=[(-65, 65), (-15, 15)], 
        maxiter=iterations,
        popsize=15
    )
    return result.x
