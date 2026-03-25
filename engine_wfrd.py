import numpy as np
import pandas as pd  # <--- IMPORTANTE: Faltaba esta línea
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

    def solve(self, iters, obs, md, inc, dip, n_layers):
        # Limpieza de datos robusta
        obs = pd.to_numeric(obs, errors='coerce')
        mask = ~np.isnan(obs)
        obs_c, md_c = obs[mask], md[mask]
        
        bounds = [(0.5, 500)] * n_layers + [(5, 40)] * (n_layers - 1) + [(1, 3)]
        
        res = differential_evolution(
            lambda m: np.sqrt(np.mean((obs_c - self.forward_model(m, md_c, inc, dip, n_layers))**2)),
            bounds=bounds, maxiter=iters, popsize=10
        )
        return res.x, res.fun

    def predict_exit(self, last_md, dttb, dtbb, future_inc, user_dip):
        # Calcula a qué distancia (MD) ocurrirá el cruce de capa
        angle_diff = np.radians(future_inc - user_dip - 90) # Ángulo relativo de ataque
        if abs(angle_diff) < 0.001: return None, "Paralelo"
        
        # Distancia al límite más cercano según tendencia
        dist_to_exit = dttb / np.sin(angle_diff) if future_inc > (90 + user_dip) else dtbb / np.sin(abs(angle_diff))
        return last_md + abs(dist_to_exit), "Techo" if future_inc > (90 + user_dip) else "Base"
