import numpy as np
from scipy.optimize import differential_evolution
from physics_engine import calculate_3d_horns

def run_proactive_inversion(measured_res, inc_val, layers):
    """Busca el mejor Shift y DIP optimizando el Misfit."""
    def objective(params):
        s_test, d_test = params
        # Usamos los datos de la capa 2 (TARGET RESERVOIR)
        rh = layers[2]['rh']
        rv = layers[2]['rv']
        synth = calculate_3d_horns(rh, rv, inc_val, d_test, s_test)
        return (np.log10(measured_res) - np.log10(synth))**2

    # Límites de búsqueda proactiva
    result = differential_evolution(objective, bounds=[(-60, 60), (-15, 15)])
    return result.x # Retorna [Best_Shift, Best_Dip]
