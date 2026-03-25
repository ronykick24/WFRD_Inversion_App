import numpy as np
from scipy.optimize import differential_evolution

class WFRD_Pro_Engine:
    def __init__(self):
        self.reach = 50.0

    def forward_model(self, m, md, inc, dip):
        # m = [R1..R5, Esp1..Esp4, Ani]
        res = m[:5]
        thick = m[5:9]
        ani = m[9]
        
        # Ángulo de ataque relativo (Crucial para el cruce de capas)
        alpha_rel = np.radians(inc - dip)
        
        # TVD proyectado: Cómo el pozo "ve" las capas según el Dip
        tvd_rel = md * np.sin(alpha_rel)
        
        # Interfaces de capas (Capa 3 es el centro del reservorio)
        z_int = np.cumsum(np.concatenate(([0], thick))) - np.sum(thick)/2
        
        response = np.full_like(md, res[0], dtype=float)
        for i in range(len(z_int)-1):
            # Transición física (Sensibilidad volumétrica de 50ft)
            weight = 0.5 * (1 + np.tanh((tvd_rel - z_int[i]) / 5.0))
            response = response * (1 - weight) + res[i+1] * weight
            
        return response * (1 + (ani - 1) * np.sin(alpha_rel)**2)

    def solve(self, obs, md, inc, dip, n_layers):
        # Ajustamos los límites según la cantidad de capas seleccionadas
        num_res = n_layers
        num_thick = n_layers - 1
        bounds = [(0.1, 500)]*num_res + [(5, 30)]*num_thick + [(1, 4)]
        
        result = differential_evolution(
            lambda m: np.sqrt(np.mean((obs - self.forward_model(self.pad_params(m, n_layers), md, inc, dip))**2)),
            bounds=bounds, popsize=10, tol=0.01
        )
        return self.pad_params(result.x, n_layers), result.fun

    def pad_params(self, m, n):
        """Asegura que siempre tengamos un vector de 10 elementos para el motor"""
        res = np.zeros(5)
        res[:n] = m[:n]
        if n < 5: res[n:] = m[n-1] # Rellena con la última resistividad
        
        thick = np.zeros(4)
        thick[:n-1] = m[n:2*n-1]
        if n < 5: thick[n-1:] = 10 # Espesor default
        
        ani = m[-1]
        return np.concatenate((res, thick, [ani], [0])) # El último es dummy dip
