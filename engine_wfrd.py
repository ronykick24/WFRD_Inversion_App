import numpy as np
from scipy.optimize import differential_evolution

class WFRD_Engine_Core:
    def forward_model(self, m, md, inc, dip, n_layers):
        res, thick, ani = m[:n_layers], m[n_layers:2*n_layers-1], m[-1]
        alpha_rel = np.radians(inc - dip)
        tvd_rel = md * np.sin(alpha_rel)
        z_int = np.cumsum(np.concatenate(([0], thick))) - np.sum(thick)/2
        
        response = np.full_like(md, res[0], dtype=float)
        for i in range(len(z_int)-1):
            weight = 0.5 * (1 + np.tanh((tvd_rel - z_int[i]) / 5.0))
            response = response * (1 - weight) + res[i+1] * weight
        return response * (1 + (ani - 1) * np.sin(alpha_rel)**2)

    def solve(self, mode, iters, obs, md, inc, dip, n_layers):
        bounds = [(0.5, 500)] * n_layers + [(5, 40)] * (n_layers - 1) + [(1, 3)]
        res = differential_evolution(
            lambda m: np.sqrt(np.mean((obs - self.forward_model(m, md, inc, dip, n_layers))**2)),
            bounds=bounds, maxiter=iters, popsize=10
        )
        return res.x, res.fun

    def predict_ahead(self, last_md, last_inc, user_dip, dist=100):
        # Proyecta la trayectoria y la capa 100ft adelante
        future_md = np.linspace(last_md, last_md + dist, 20)
        # Trayectoria proyectada (asumiendo inc constante)
        rel_path = (future_md - last_md) * np.sin(np.radians(last_inc - 90))
        # Capas proyectadas (siguiendo el DIP)
        rel_layer = -(future_md - last_md) * np.tan(np.radians(user_dip))
        return future_md, rel_path, rel_layer
