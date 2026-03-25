import numpy as np
from scipy.optimize import differential_evolution
from physics_engine import calculate_3d_horns

def run_proactive_inversion(res_vals, inc_vals, layers, iterations=1000):
    """Inversión Estocástica de alta intensidad para multicapa."""
    res_vals = np.array(res_vals, dtype=float)
    inc_vals = np.array(inc_vals, dtype=float)
    mask = ~np.isnan(res_vals)
    r_c, i_c = res_vals[mask], inc_vals[mask]

    def objective(params):
        s_test, d_test = params
        # Enfocado en la capa del reservorio (Capa 2)
        rh, rv = float(layers[2]['rh']), float(layers[2]['rv'])
        
        # Misfit determinístico
        errors = []
        for j in range(len(r_c)):
            synth = calculate_3d_horns(rh, rv, i_c[j], d_test, s_test)
            errors.append((np.log10(r_c[j] + 1e-6) - np.log10(synth + 1e-6))**2)
        return np.mean(errors)

    # Configuración de 1000 iteraciones máx para precisión proactiva
    result = differential_evolution(
        objective, 
        bounds=[(-70, 70), (-20, 20)], 
        maxiter=iterations,
        popsize=15,
        mutation=(0.5, 1),
        recombination=0.7
    )
    return result.x
