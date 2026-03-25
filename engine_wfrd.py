import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, least_squares
from physics_engine import calculate_3d_anisotropy, get_perpendicular_distance

class WFRD_Engine_Core:
    def forward_model(self, m, md, inc, dip, n_layers):
        if len(md) == 0: return np.array([])
        res_h = np.clip(m[:n_layers], 0.1, 1000)
        thick = np.clip(m[n_layers:2*n_layers-1], 1, 100)
        ani_ratio = np.clip(m[-1], 1.0, 5.0) 
        
        tvd_perp = get_perpendicular_distance(md, inc, dip)
        z_int = np.cumsum(np.concatenate(([0], thick))) - np.sum(thick)/2
        
        rh_layer = np.full_like(md, res_h[0], dtype=float)
        for i in range(len(z_int)-1):
            weight = 0.5 * (1 + np.tanh(np.clip((tvd_perp - z_int[i])/5, -20, 20)))
            rh_layer = rh_layer * (1 - weight) + res_h[i+1] * weight
            
        rv_layer = rh_layer * ani_ratio
        return calculate_3d_anisotropy(rh_layer, rv_layer, inc, dip)

    def solve(self, mode, obs, md, inc, dip, n_layers):
        obs_np = pd.to_numeric(obs, errors='coerce').values
        md_np = pd.to_numeric(md, errors='coerce').values
        mask = np.isfinite(obs_np) & np.isfinite(md_np)
        obs_c, md_c = obs_np[mask], md_np[mask]
        
        bounds = [(0.2, 1000)] * n_layers + [(2, 50)] * (n_layers - 1) + [(1.0, 5.0)]
        
        def objective(m):
            try:
                pred = self.forward_model(m, md_c, inc, dip, n_layers)
                return np.sqrt(np.mean((obs_c - pred)**2)) if not np.any(np.isnan(pred)) else 1e12
            except: return 1e12

        if "Global" in mode:
            res = differential_evolution(objective, bounds=bounds, maxiter=1000, popsize=15, polish=False)
            return res.x, res.fun
        elif "Local" in mode:
            res = differential_evolution(objective, bounds=bounds, maxiter=100, popsize=10, polish=False)
            return res.x, res.fun
        else:
            x0 = [10]*n_layers + [15]*(n_layers-1) + [1.5]
            sol = least_squares(lambda m: obs_c - self.forward_model(m, md_c, inc, dip, n_layers), 
                              x0=x0, bounds=([b[0] for b in bounds], [b[1] for b in bounds]), max_nfev=100)
            return sol.x, np.sqrt(np.mean(sol.fun**2))
