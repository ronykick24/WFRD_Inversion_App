import numpy as np
from scipy.optimize import differential_evolution
from physics_engine import calculate_3d_horns

def run_proactive_inversion(res_vals, inc_vals, layers, progress_bar):
    """Ejecuta la inversión convirtiendo datos a float para evitar TypeError."""
    # Forzar conversión a float y limpiar NaNs
    res_vals = np.array(res_vals, dtype=float)
    inc_vals = np.array(inc_vals, dtype=float)
    
    mask = ~np.isnan(res_vals)
    r_c, i_c = res_vals[mask], inc_vals[mask]

    def objective(params):
        s_test, d_test = params
        rh, rv = float(layers[2]['rh']), float(layers[2]['rv'])
        errors = []
        for j in range(len(r_c)):
            # Distancia relativa para el cálculo del error
            synth = calculate_3d_horns(rh, rv, i_c[j], d_test, s_test)
            errors.append((np.log10(r_c[j] + 1e-6) - np.log10(synth + 1e-6))**2)
        return np.mean(errors)

    progress_bar.progress(50, text="Iterando modelos 3D...")
    result = differential_evolution(objective, bounds=[(-60, 60), (-15, 15)], tol=0.01)
    progress_bar.progress(100, text="Inversión Completada")
    return result.x
