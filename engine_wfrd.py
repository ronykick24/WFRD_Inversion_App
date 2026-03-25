import numpy as np
from scipy.optimize import differential_evolution

class WFRD_Engine_Core:
    def forward_model(self, m, md, inc, dip, n_layers):
        res, thick, ani = m[:n_layers], m[n_layers:2*n_layers-1], m[-1]
        # Ángulo relativo: Inc (trayectoria) vs Dip (formación)
        alpha_rel = np.radians(inc - dip)
        tvd_rel = md * np.sin(alpha_rel)
        
        z_int = np.cumsum(np.concatenate(([0], thick))) - np.sum(thick)/2
        
        response = np.full_like(md, res[0], dtype=float)
        for i in range(len(z_int)-1):
            weight = 0.5 * (1 + np.tanh((tvd_rel - z_int[i]) / 5.0))
            response = response * (1 - weight) + res[i+1] * weight
        return response * (1 + (ani - 1) * np.sin(alpha_rel)**2)

    def solve(self, iters, obs, md, inc, dip, n_layers):
        # Asegurar que obs sea numérico y sin NaNs
        obs = pd.to_numeric(obs, errors='coerce')
        mask = ~np.isnan(obs)
        obs_c, md_c = obs[mask], md[mask]
        
        bounds = [(0.5, 500)] * n_layers + [(5, 40)] * (n_layers - 1) + [(1, 3)]
        
        res = differential_evolution(
            lambda m: np.sqrt(np.mean((obs_c - self.forward_model(m, md_c, inc, dip, n_layers))**2)),
            bounds=bounds, maxiter=iters, popsize=10
        )
        return res.x, res.fun

    def predict_ahead(self, last_md, last_inc, future_inc, user_dip, dist=100):
        f_md = np.linspace(last_md, last_md + dist, 30)
        # Simulación de trayectoria con la inclinación proyectada
        rel_path = (f_md - last_md) * np.cos(np.radians(future_inc)) # Cambio a TVD proyectado
        # Mapeo de formación siguiendo el DIP
        rel_layer = -(f_md - last_md) * np.tan(np.radians(user_dip))
        return f_md, rel_path, rel_layer
