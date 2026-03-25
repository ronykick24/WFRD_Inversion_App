import numpy as np
from scipy.optimize import differential_evolution

class WFRD_Pro_Simulator:
    def __init__(self):
        self.reach = 50.0

    def forward_model_2D(self, m, md, inc):
        # m: [R1, R2, R3, R4, R5, Esp1, Esp2, Esp3, Esp4, Ani, Dip]
        res_layers = m[:5]
        thicknesses = m[5:9]
        ani_ratio = m[9]
        dip = m[10]
        
        alpha_rel = np.radians(inc - dip)
        tvd_rel = md * np.sin(alpha_rel)
        
        z_interfaces = np.cumsum(np.concatenate(([0], thicknesses))) - np.sum(thicknesses)/2
        
        response = np.full_like(md, res_layers[0], dtype=float)
        for i in range(len(z_interfaces)-1):
            # Transición física de la onda (Sensibilidad de 50ft)
            weight = 0.5 * (1 + np.tanh((tvd_rel - z_interfaces[i]) / 6.0))
            response = response * (1 - weight) + res_layers[i+1] * weight
            
        return response * (1 + (ani_ratio - 1) * np.sin(alpha_rel)**2)

    def solve(self, obs_data, md_array, inc, dip_manual):
        # Inversión Estocástica acotada por el Dip manual del usuario
        bounds = [(0.1, 500)]*5 + [(5, 30)]*4 + [(1, 4)] + [(dip_manual-2, dip_manual+2)]
        result = differential_evolution(
            lambda m: np.sqrt(np.mean((obs_data - self.forward_model_2D(m, md_array, inc))**2)),
            bounds=bounds, popsize=8, tol=0.01
        )
        return result.x, result.fun
