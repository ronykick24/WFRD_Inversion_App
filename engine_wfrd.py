import numpy as np
from scipy.optimize import differential_evolution
from physics_engine import calculate_3d_horns

def run_proactive_inversion(measured_res_array, md_array, inc_array, layers):
    """Algoritmo Estocástico: Encuentra el Shift y DIP óptimos usando datos reales."""
    def objective(params):
        s_test, d_test = params
        rh, rv = layers[2]['rh'], layers[2]['rv'] # Capa Reservorio
        
        misfit_points = []
        for i in range(len(measured_res_array)):
            synth = calculate_3d_horns(rh, rv, inc_array[i], d_test, s_test)
            misfit_points.append((np.log10(measured_res_array[i]) - np.log10(synth))**2)
            
        return np.mean(misfit_points)

    result = differential_evolution(objective, bounds=[(-60, 60), (-15, 15)])
    return result.x
