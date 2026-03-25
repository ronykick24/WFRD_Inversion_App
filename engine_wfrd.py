import numpy as np
from scipy.optimize import differential_evolution

class WFRD_Engine_Core:
    def __init__(self):
        self.reach = 50.0

    def forward_model(self, m, md, inc, dip):
        # m = [R1..R5, E1..E4, Ani]
        res, thick, ani = m[:5], m[5:9], m[9]
        alpha_rel = np.radians(inc - dip)
        tvd_rel = md * np.sin(alpha_rel)
        
        # Interfaces relativas al centro del reservorio (Capa 4)
        z_int = np.cumsum(np.concatenate(([0], thick))) - np.sum(thick[:3]) 
        
        response = np.full_like(md, res[0], dtype=float)
        for i in range(len(z_int)-1):
            weight = 0.5 * (1 + np.tanh((tvd_rel - z_int[i]) / 5.0))
            response = response * (1 - weight) + res[i+1] * weight
            
        return response * (1 + (ani - 1) * np.sin(alpha_rel)**2)

    def solve(self, mode, iters, obs, md, inc, dip):
        # Secuencia: Sello1, Arena1, Sello2, Reservorio(PAY), Sello3
        bounds = [(1, 5), (10, 40), (1, 5), (40, 500), (1, 5)] + [(10, 30)]*4 + [(1, 2.5)]
        res = differential_evolution(
            lambda m: np.sqrt(np.mean((obs - self.forward_model(m, md, inc, dip))**2)),
            bounds=bounds, maxiter=iters, popsize=8
        )
        return res.x, res.fun
