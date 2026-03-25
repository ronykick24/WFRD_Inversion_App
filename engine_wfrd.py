import numpy as np
from scipy.optimize import differential_evolution
from physics_engine import calculate_3d_horns

def run_proactive_inversion(res_vals, inc_vals, layers, progress_bar):
    """Ejecuta la inversión con visualización de progreso."""
    mask = ~np.isnan(res_vals)
    r_c, i_c = res_vals[mask], inc_vals[mask]

    def objective(params):
        s_test, d_test = params
        rh, rv = layers[2]['rh'], layers[2]['rv']
        errors = []
        for j in range(len(r_c)):
            synth = calculate_3d_horns(rh, rv, i_c[j], d_test, s_test)
            errors.append((np.log10(r_c[j] + 1e-6) - np.log10(synth + 1e-6))**2)
        return np.mean(errors)

    # Simulación de iteraciones para el usuario
    progress_bar.progress(50, text="Iterando modelos estocásticos...")
    result = differential_evolution(objective, bounds=[(-60, 60), (-15, 15)], tol=0.01)
    progress_bar.progress(100, text="Inversión Completada")
    return result.x
