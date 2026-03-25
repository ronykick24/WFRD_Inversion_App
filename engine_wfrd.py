import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

class WFRD_Engine_Core:
    def forward_model(self, m, md, inc, dip, n_layers):
        if len(md) == 0: return np.array([])
        res, thick, ani = m[:n_layers], m[n_layers:2*n_layers-1], m[-1]
        
        # Geometría de Mapeo: Trayectoria vs Formación
        alpha_rel = np.radians(inc - dip)
        tvd_rel = md * np.sin(alpha_rel)
        z_int = np.cumsum(np.concatenate(([0], thick))) - np.sum(thick)/2
        
        response = np.full_like(md, res[0], dtype=float)
        for i in range(len(z_int)-1):
            weight = 0.5 * (1 + np.tanh((tvd_rel - z_int[i]) / 5.0))
            response = response * (1 - weight) + res[i+1] * weight
        return response * (1 + (ani - 1) * np.sin(alpha_rel)**2)

    def solve(self, iters, obs, md, inc, dip, n_layers):
        # Conversión forzada a arreglos numéricos de NumPy
        obs_np = pd.to_numeric(obs, errors='coerce').values
        md_np = pd.to_numeric(md, errors='coerce').values
        
        mask = ~np.isnan(obs_np) & ~np.isnan(md_np)
        obs_c, md_c = obs_np[mask], md_np[mask]
        
        if len(obs_c) < 5:
            return np.array([10]*n_layers + [15]*(n_layers-1) + [1.0]), 0.0

        bounds = [(0.5, 500)] * n_layers + [(5, 40)] * (n_layers - 1) + [(1, 3)]
        
        res = differential_evolution(
            lambda m: np.sqrt(np.mean((obs_c - self.forward_model(m, md_c, inc, dip, n_layers))**2)),
            bounds=bounds, maxiter=iters, popsize=10
        )
        return res.x, res.fun

    def predict_exit(self, last_md, dttb, dtbb, future_inc, user_dip):
        # Ángulo relativo de ataque respecto a la formación
        rel_attack = future_inc - (90 + user_dip)
        if abs(rel_attack) < 0.1: return None, "Paralelo"
        
        rad = np.radians(rel_attack)
        if rel_attack > 0: # Subiendo hacia el techo
            dist = dttb / np.sin(rad)
            return last_md + abs(dist), "TECHO"
        else: # Bajando hacia la base
            dist = dtbb / np.sin(abs(rad))
            return last_md + abs(dist), "BASE"
