import numpy as np
from scipy.optimize import differential_evolution, least_squares

class WFRD_Engine_Core:
    def __init__(self):
        self.reach = 50.0

    def forward_model(self, m, md, inc, dip):
        # m = [R1..R5, Esp1..Esp4, Ani]
        res = m[:5]
        thick = m[5:9]
        ani = m[9]
        
        # Lógica Geofísica: NBI y Proyección de Capas
        # Dip negativo -> Capas suben (+ TVD relativo)
        # Dip positivo -> Capas bajan (- TVD relativo)
        alpha_rel = np.radians(inc - dip)
        tvd_rel = md * np.sin(alpha_rel)
        
        z_int = np.cumsum(np.concatenate(([0], thick))) - np.sum(thick)/2
        
        response = np.full_like(md, res[0], dtype=float)
        for i in range(len(z_int)-1):
            weight = 0.5 * (1 + np.tanh((tvd_rel - z_int[i]) / 5.0))
            response = response * (1 - weight) + res[i+1] * weight
            
        return response * (1 + (ani - 1) * np.sin(alpha_rel)**2)

    def solve(self, mode, obs, md, inc, dip, n_layers, iters):
        bounds = [(0.1, 500)]*n_layers + [(5, 25)]*(n_layers-1) + [(1, 4)]
        
        if mode == "Estocástico":
            res = differential_evolution(
                lambda m: np.sqrt(np.mean((obs - self.forward_model(self.pad(m, n_layers), md, inc, dip))**2)),
                bounds=bounds, maxiter=iters, popsize=10
            )
            return self.pad(res.x, n_layers), res.fun
        else:
            x0 = [10]*n_layers + [12]*(n_layers-1) + [1.5]
            res = least_squares(
                lambda m: obs - self.forward_model(self.pad(m, n_layers), md, inc, dip),
                x0=x0, bounds=([b[0] for b in bounds], [b[1] for b in bounds]), max_nfev=iters
            )
            return self.pad(res.x, n_layers), np.mean(res.fun**2)

    def pad(self, m, n):
        res, thick = np.zeros(5), np.zeros(4)
        res[:n], thick[:n-1] = m[:n], m[n:2*n-1]
        if n < 5: res[n:], thick[n-1:] = m[n-1], 10
        return np.concatenate((res, thick, [m[-1]]))
