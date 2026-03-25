import numpy as np
from scipy.optimize import differential_evolution

class WFRD_Engine_Core:
    def __init__(self):
        pass

    def forward_model(self, m, md, inc, dip, n_layers):
        # m estructurado: [Resistividades, Espesores, Anisotropía]
        res = m[:n_layers]
        thick = m[n_layers:2*n_layers-1]
        ani = m[-1]
        
        alpha_rel = np.radians(inc - dip)
        tvd_rel = md * np.sin(alpha_rel)
        
        # Interfaces centradas en la capa objetivo (capa central)
        z_int = np.cumsum(np.concatenate(([0], thick))) - np.sum(thick)/2
        
        response = np.full_like(md, res[0], dtype=float)
        for i in range(len(z_int)-1):
            weight = 0.5 * (1 + np.tanh((tvd_rel - z_int[i]) / 5.0))
            response = response * (1 - weight) + res[i+1] * weight
            
        return response * (1 + (ani - 1) * np.sin(alpha_rel)**2)

    def solve(self, iters, obs, md, inc, dip, n_layers):
        # Definición de límites según litología (Sello -> Reservorio -> Sello)
        res_bounds = [(0.5, 10)] * n_layers
        # Forzamos que la capa central sea la más resistiva (Reservorio)
        res_bounds[n_layers // 2] = (50, 500) 
        
        thick_bounds = [(5, 30)] * (n_layers - 1)
        bounds = res_bounds + thick_bounds + [(1, 3)]
        
        res = differential_evolution(
            lambda m: np.sqrt(np.mean((obs - self.forward_model(m, md, inc, dip, n_layers))**2)),
            bounds=bounds, maxiter=iters, popsize=10
        )
        return res.x, res.fun
