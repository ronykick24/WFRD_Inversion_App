import numpy as np
from scipy.optimize import differential_evolution, least_squares
from physics_engine import calculate_3d_anisotropy, apply_polarization_horns

class WFRD_Engine_Core:
    def forward_model(self, m, md, inc, dip, n_layers):
        res_h = np.clip(m[:n_layers], 0.1, 1000)
        thick = np.clip(m[n_layers:2*n_layers-1], 1, 100)
        ani_ratio = m[-1]
        
        alpha_rel = np.radians(inc - (90 + dip))
        tvd_perp = md * np.sin(alpha_rel)
        z_int = np.cumsum(np.concatenate(([0], thick))) - np.sum(thick)/2
        
        rh_layer = np.full_like(md, res_h[0], dtype=float)
        for i in range(len(z_int)-1):
            weight = 0.5 * (1 + np.tanh((tvd_perp - z_int[i])/4.0))
            rh_layer = rh_layer * (1 - weight) + res_h[i+1] * weight
            
        rv_layer = rh_layer * ani_ratio
        ra = calculate_3d_anisotropy(rh_layer, rv_layer, inc, dip)
        # Añadimos física de cuernos de polarización
        return apply_polarization_horns(ra, md, z_int, tvd_perp)

    def solve(self, mode, obs, md, inc, dip, n_layers):
        obs_np = np.nan_to_num(np.asarray(obs, dtype=float), nan=10.0)
        md_np = np.asarray(md, dtype=float)
        bounds = [(0.2, 1000)] * n_layers + [(2, 60)] * (n_layers - 1) + [(1.0, 4.0)]
        
        def objective(m):
            try:
                pred = self.forward_model(m, md_np, inc, dip, n_layers)
                return np.sqrt(np.mean((np.log10(obs_np) - np.log10(pred))**2))
            except: return 1e12

        if "Global" in mode:
            res = differential_evolution(objective, bounds=bounds, maxiter=1000, popsize=15)
        else:
            x0 = [10]*n_layers + [15]*(n_layers-1) + [1.5]
            res = least_squares(lambda m: np.log10(obs_np) - np.log10(self.forward_model(m, md_np, inc, dip, n_layers)), 
                                x0=x0, bounds=([b[0] for b in bounds], [b[1] for b in bounds]), max_nfev=100)
        return res.x, 0.0
