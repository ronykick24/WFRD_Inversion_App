import numpy as np
from scipy.optimize import differential_evolution

class WFRD_Advanced_Engine:
    def __init__(self):
        self.reach = 50.0

    def forward_model(self, m, md, inc):
        # m = [R1..R5, Esp1..Esp4, Ani, Dip] (11 parámetros)
        res = m[:5]
        thick = m[5:9]
        ani = m[9]
        dip = m[10]
        
        rel_angle = np.radians(inc - dip)
        tvd_rel = md * np.sin(rel_angle)
        
        # Fronteras de capas centradas
        z_int = np.cumsum(np.concatenate(([0], thick))) - np.sum(thick)/2
        
        # Respuesta con transición física (shading real)
        response = np.full_like(md, res[0], dtype=float)
        for i in range(len(z_int)-1):
            # Simulación de la física de inducción de 50ft
            weight = 0.5 * (1 + np.tanh((tvd_rel - z_int[i]) / 5.5))
            response = response * (1 - weight) + res[i+1] * weight
            
        return response * (1 + (ani - 1) * np.sin(rel_angle)**2)

    def solve(self, obs, md, inc, dip_hint):
        # Buscamos en un rango cercano al dip que el usuario mueve en el slider
        bounds = [(0.1, 500)]*5 + [(5, 25)]*4 + [(1, 4)] + [(dip_hint-3, dip_hint+3)]
        res = differential_evolution(
            lambda m: np.sqrt(np.mean((obs - self.forward_model(m, md, inc))**2)),
            bounds=bounds, popsize=8, tol=0.01
        )
        return res.x, res.fun
