import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, least_squares

class WFRD_Engine_Core:
    def forward_model(self, m, md, inc, dip, n_layers):
        if len(md) == 0: return np.array([])
        
        # Estructura del modelo: [Resistividades, Espesores, Anisotropía Lambda]
        res = np.clip(m[:n_layers], 0.1, 1000)
        thick = np.clip(m[n_layers:2*n_layers-1], 1, 100)
        lambda_ani = m[-1] # Relación Rv/Rh
        
        # Ángulo de ataque relativo (Crucial para el DTB real)
        # alpha_rel es el ángulo entre la trayectoria y la cara de la capa
        alpha_rel = np.radians(inc - (90 + dip)) 
        
        # TVD relativo perpendicular a la formación (Mapeo 2D)
        tvd_perp = md * np.sin(alpha_rel)
        
        z_int = np.cumsum(np.concatenate(([0], thick))) - np.sum(thick)/2
        
        # Cálculo de la resistividad aparente con anisotropía
        # Ra = Rh / sqrt(cos^2(theta) + lambda^2 * sin^2(theta))
        theta_res = np.radians(inc - dip) # Ángulo respecto al eje de anisotropía
        ani_factor = 1.0 / np.sqrt(np.cos(theta_res)**2 + (lambda_ani**2) * np.sin(theta_res)**2)
        
        response = np.full_like(md, res[0], dtype=float)
        for i in range(len(z_int)-1):
            diff = np.clip((tvd_perp - z_int[i]) / 5.0, -20, 20)
            weight = 0.5 * (1 + np.tanh(diff))
            response = response * (1 - weight) + res[i+1] * weight
            
        return response * ani_factor

    def solve(self, mode, obs, md, inc, dip, n_layers):
        obs_np = pd.to_numeric(obs, errors='coerce').values
        md_np = pd.to_numeric(md, errors='coerce').values
        mask = np.isfinite(obs_np) & np.isfinite(md_np)
        obs_c, md_c = obs_np[mask], md_np[mask]
        
        if len(obs_c) < 5: return np.array([10]*n_layers + [15]*(n_layers-1) + [1.5]), 0.0

        # Bounds: Res [0.2-1000], Thick [2-50], Anisotropía Lambda [1.0-4.0]
        bounds = [(0.2, 1000)] * n_layers + [(2, 50)] * (n_layers - 1) + [(1.0, 4.0)]
        x0 = [10]*n_layers + [15]*(n_layers-1) + [1.5]

        def obj_func(m):
            return np.sqrt(np.mean(np.square(obs_c - self.forward_model(m, md_c, inc, dip, n_layers))))

        if "Global" in mode:
            res = differential_evolution(obj_func, bounds=bounds, maxiter=1000, popsize=15, polish=True)
        elif "Local" in mode:
            res = differential_evolution(obj_func, bounds=bounds, maxiter=100, popsize=10, polish=False)
        else:
            lb, ub = [b[0] for b in bounds], [b[1] for b in bounds]
            sol = least_squares(lambda m: obs_c - self.forward_model(m, md_c, inc, dip, n_layers), 
                              x0=x0, bounds=(lb, ub), max_nfev=100)
            return sol.x, np.sqrt(np.mean(sol.fun**2))
        
        return res.x, res.fun

    def predict_exit(self, last_md, dttb, dtbb, sim_inc, user_dip):
        # El ángulo de aproximación real considera la inclinación de la capa
        rel_attack = sim_inc - (90 + user_dip)
        if abs(rel_attack) < 0.1: return None, "Paralelo"
        rad = np.radians(rel_attack)
        dist = dttb / np.sin(rad) if rel_attack > 0 else dtbb / np.sin(abs(rad))
        return last_md + abs(dist), "TECHO" if rel_attack > 0 else "BASE"
