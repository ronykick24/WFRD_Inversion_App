import numpy as np
from scipy.optimize import differential_evolution
from physics_engine import calculate_forward_model

def run_ahta_inversion(res_data, inc_data, layers, iterations=1000):
    """Encuentra el mejor DIP y DTBss minimizando el error logarítmico."""
    res_vals = np.array(res_data, dtype=float)
    inc_vals = np.array(inc_data, dtype=float)
    mask = ~np.isnan(res_vals)
    r_c, i_c = res_vals[mask], inc_vals[mask]

    def objective(params):
        shift_t, dip_t = params
        # Capa objetivo (Reservorio) definida en el índice 2 de layers
        rh, rv = layers[2]['rh'], layers[2]['rv']
        
        errors = []
        for j in range(len(r_c)):
            # Distancia estructural calculada para cada punto de la muestra
            d_inst = shift_t + (j * np.tan(np.radians(dip_t)))
            synth = calculate_forward_model(rh, rv, i_c[j], dip_t, d_inst)
            # Misfit logarítmico para manejar rangos de resistividad
            errors.append((np.log10(r_c[j]+1e-6) - np.log10(synth+1e-6))**2)
        return np.mean(errors)

    # Optimización estocástica global
    res = differential_evolution(objective, bounds=[(-65, 65), (-15, 15)], maxiter=iterations)
    return res.x # [Shift, Dip]
