import numpy as np
from scipy.optimize import differential_evolution
from physics_engine import calculate_3d_horns

def run_proactive_inversion(res_vals, inc_vals, layers):
    """Encuentra el mejor Shift/DIP minimizando el error."""
    def objective(params):
        s_test, d_test = params
        rh, rv = layers[2]['rh'], layers[2]['rv'] # Capa Reservorio
        
        # Calculamos el error promedio en la ventana de datos
        errors = []
        for i in range(len(res_vals)):
            synth = calculate_3d_horns(rh, rv, inc_vals[i], d_test, s_test)
            errors.append((np.log10(res_vals[i]) - np.log10(synth))**2)
        return np.mean(errors)

    result = differential_evolution(objective, bounds=[(-60, 60), (-15, 15)])
    return result.x
