import numpy as np
from scipy.optimize import differential_evolution
from physics_engine import calculate_3d_horns

def run_proactive_inversion(measured_res, inc_val, layers):
    """Algoritmo Estocástico: Encuentra el Shift y DIP óptimos."""
    def objective(params):
        s_test, d_test = params
        # Capa 2 es el TARGET RESERVOIR (100 ohmm)
        rh, rv = layers[2]['rh'], layers[2]['rv']
        synth = calculate_3d_horns(rh, rv, inc_val, d_test, s_test)
        # Misfit logarítmico
        return (np.log10(measured_res) - np.log10(synth))**2

    result = differential_evolution(objective, bounds=[(-60, 60), (-15, 15)])
    return result.x # Retorna [Best_Shift, Best_Dip]
