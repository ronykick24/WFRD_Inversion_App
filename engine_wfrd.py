import numpy as np
from scipy.optimize import differential_evolution

class WFRD_Engine_Core:
    def forward_model(self, m, md, inc, dip, n_layers):
        # m: [Resistividades, Espesores, Anisotropía]
        res = m[:n_layers]
        thick = m[n_layers:2*n_layers-1]
        ani = m[-1]
        
        # Ángulo relativo corregido para mapeo
        alpha_rel = np.radians(inc - dip)
        # TVD relativo al centro del modelo de capas
        tvd_rel = md * np.sin(alpha_rel)
        
        # Definición de fronteras (fijas en el modelo tierra)
        z_int = np.cumsum(np.concatenate(([0], thick))) - np.sum(thick)/2
        
        # Simulación de respuesta
        response = np.full_like(md, res[0], dtype=float)
        for i in range(len(z_int)-1):
            # Aseguramos que el cálculo sea estable
            weight = 0.5 * (1 + np.tanh((tvd_rel - z_int[i]) / 5.0))
            response = response * (1 - weight) + res[i+1] * weight
            
        return response * (1 + (ani - 1) * np.sin(alpha_rel)**2)

    def solve(self, mode, iters, obs, md, inc, dip, n_layers):
        # Limpieza de seguridad para evitar el RuntimeError
        mask = ~np.isnan(obs)
        obs_clean = obs[mask]
        md_clean = md[mask]
        
        # Bounds: [Res1..ResN, Thick1..ThickN-1, Ani]
        bounds = [(0.5, 500)] * n_layers + [(5, 40)] * (n_layers - 1) + [(1, 3)]
        
        # La función lambda ahora usa los datos limpios y el n_layers correcto
        def objective(m):
            pred = self.forward_model(m, md_clean, inc, dip, n_layers)
            return np.sqrt(np.mean((obs_clean - pred)**2))

        res = differential_evolution(objective, bounds=bounds, maxiter=iters, popsize=10)
        return res.x, res.fun

    def predict_ahead(self, last_md, last_inc, user_dip, dist=100):
        # Simulación de 100 pies adelante
        future_md = np.linspace(last_md, last_md + dist, 30)
        # Trayectoria proyectada (TVD relativo acumulado)
        rel_path = (future_md - last_md) * np.sin(np.radians(last_inc - 90))
        # Formación proyectada (Mapeo por DIP)
        rel_layer_shift = -(future_md - last_md) * np.tan(np.radians(user_dip))
        return future_md, rel_path, rel_layer_shift
