import numpy as np
from scipy.optimize import differential_evolution, least_squares

class WFRD_Engine_Core:
    def __init__(self):
        self.reach = 50.0

    def forward_model(self, m, md, inc, dip):
        # m = [R1..R5, Esp1..Esp4, Ani]
        res, thick, ani = m[:5], m[5:9], m[9]
        
        # Geometría: Dip (-) -> Capas ascendentes (+ TVD relativo)
        alpha_rel = np.radians(inc - dip)
        tvd_rel = md * np.sin(alpha_rel)
        
        # Definición de fronteras
        z_int = np.cumsum(np.concatenate(([0], thick))) - np.sum(thick)/2
        
        # Cálculo de respuesta de herramienta (Simulación de inducción)
        response = np.full_like(md, res[0], dtype=float)
        for i in range(len(z_int)-1):
            weight = 0.5 * (1 + np.tanh((tvd_rel - z_int[i]) / 5.0))
            response = response * (1 - weight) + res[i+1] * weight
            
        return response * (1 + (ani - 1) * np.sin(alpha_rel)**2)

    def solve(self, mode, iters, obs, md, inc, dip, n_layers):
        # Límites: Res (0.1-500), Espesores (5-30ft), Anisotropía (1-4)
        bounds = [(0.1, 500)]*n_layers + [(5, 35)]*(n_layers-1) + [(1, 4)]
        
        if mode == "Estocástico (Global)":
            res = differential_evolution(
                lambda m: np.sqrt(np.mean((obs - self.forward_model(self.pad(m, n_layers), md, inc, dip))**2)),
                bounds=bounds, maxiter=iters, popsize=10
            )
            return self.pad(res.x, n_layers), res.fun
        else:
            # Determinístico (Local / Least Squares)
            x0 = [10]*n_layers + [15]*(n_layers-1) + [1.5]
            lb = [b[0] for b in bounds]
            ub = [b[1] for b in bounds]
            res = least_squares(
                lambda m: obs - self.forward_model(self.pad(m, n_layers), md, inc, dip),
                x0=x0, bounds=(lb, ub), max_nfev=iters
            )
            return self.pad(res.x, n_layers), np.mean(res.fun**2)

    def pad(self, m, n):
        res, thick = np.zeros(5), np.zeros(4)
        res[:n], thick[:n-1] = m[:n], m[n:2*n-1]
        if n < 5: res[n:], thick[n-1:] = m[n-1], 10
        return np.concatenate((res, thick, [m[-1]]))
