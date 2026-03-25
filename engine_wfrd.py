import numpy as np
from scipy.optimize import differential_evolution

class WFRD_Simulator:
    def __init__(self):
        self.reach = 50.0

    def forward_model_2D(self, m, md, inc):
        # m contiene: [R1, R2, R3, R4, R5, Esp1, Esp2, Esp3, Esp4, Ani, Dip] (Total 11)
        res_layers = m[:5]
        thicknesses = m[5:9]
        ani_ratio = m[9]
        dip = m[10]

        # Ángulo relativo pozo-capa (Crucial para 85° y sensibilidad)
        alpha_rel = np.radians(inc - dip)
        
        # Proyección TVD respecto al buzamiento
        tvd_rel = md * np.sin(alpha_rel)
        
        # Definir fronteras de capas (Interfaces)
        # Centramos las 5 capas respecto al pozo
        z_interfaces = np.cumsum(np.concatenate(([0], thicknesses))) - np.sum(thicknesses)/2
        
        # Simulación de respuesta 1D/2D mediante promedio pesado por distancia
        # Representa cómo la herramienta "siente" las capas antes de cruzarlas
        response = np.full_like(md, res_layers[0], dtype=float)
        for i in range(len(z_interfaces)-1):
            # La física de la herramienta GuideWave: Sensibilidad volumétrica (tanh)
            weight = 0.5 * (1 + np.tanh((tvd_rel - z_interfaces[i]) / 5.0))
            response = response * (1 - weight) + res_layers[i+1] * weight
        
        # Aplicar efecto de anisotropía (Rh vs Rv) a alto ángulo
        return response * (1 + (ani_ratio - 1) * np.sin(alpha_rel)**2)

    def solve(self, obs_data, md_array, inc):
        # Bounds: 5 Resists (0.1-500), 4 Thicks (5-30ft), 1 Ani (1-4), 1 Dip (-10 a 10)
        bounds = [(0.1, 500)]*5 + [(5, 30)]*4 + [(1, 4)] + [(-10, 10)]
        
        result = differential_evolution(
            lambda m: np.sqrt(np.mean((obs_data - self.forward_model_2D(m, md_array, inc))**2)),
            bounds=bounds, popsize=10, mutation=(0.5, 1), recombination=0.7, tol=0.01
        )
        return result.x, result.fun
