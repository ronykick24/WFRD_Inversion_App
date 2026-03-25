import numpy as np
from scipy.optimize import differential_evolution, least_squares, minimize

class WFRD_Engine_Core:
    def __init__(self):
        pass

    def forward_model(self, m, md, inc, dip, n_layers):
        res = m[:n_layers]
        thick = m[n_layers:2*n_layers-1]
        ani = m[-1]
        
        alpha_rel = np.radians(inc - dip)
        tvd_rel = md * np.sin(alpha_rel)
        
        # Interfaces: Capa objetivo es la central (n_layers // 2)
        z_int = np.cumsum(np.concatenate(([0], thick))) - np.sum(thick)/2
        
        response = np.full_like(md, res[0], dtype=float)
        for i in range(len(z_int)-1):
            weight = 0.5 * (1 + np.tanh((tvd_rel - z_int[i]) / 5.0))
            response = response * (1 - weight) + res[i+1] * weight
            
        return response * (1 + (ani - 1) * np.sin(alpha_rel)**2)

    def solve(self, mode, iters, obs, md, inc, dip, n_layers):
        # Configuración de límites por capas
        res_bounds = [(0.5, 500)] * n_layers
        thick_bounds = [(5, 40)] * (n_layers - 1)
        bounds = res_bounds + thick_bounds + [(1, 3)]
        x0 = [10]*n_layers + [15]*(n_layers-1) + [1.5]

        if mode == "Estocástico Global":
            # Differential Evolution: Explora todo el espacio
            res = differential_evolution(
                lambda m: np.sqrt(np.mean((obs - self.forward_model(m, md, inc, dip, n_layers))**2)),
                bounds=bounds, maxiter=iters, popsize=12
            )
            return res.x, res.fun

        elif mode == "Estocástico Local":
            # Dual Annealing o similar (aquí simplificado con minimize estocástico)
            res = minimize(
                lambda m: np.sqrt(np.mean((obs - self.forward_model(m, md, inc, dip, n_layers))**2)),
                x0=x0, bounds=bounds, method='L-BFGS-B', options={'maxiter': iters}
            )
            return res.x, res.fun

        else: # Determinístico
            lb, ub = [b[0] for b in bounds], [b[1] for b in bounds]
            res = least_squares(
                lambda m: obs - self.forward_model(m, md, inc, dip, n_layers),
                x0=x0, bounds=(lb, ub), max_nfev=iters
            )
            return res.x, np.mean(res.fun**2)
