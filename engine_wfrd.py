import numpy as np
from scipy.optimize import differential_evolution, least_squares

class WFRD_Engine_Core:
    def __init__(self):
        self.reach = 50.0

    def forward_model(self, m, md, inc, dip):
        # m = [R1, R2, R3, R4, R5, E1, E2, E3, E4, Ani]
        res, thick, ani = m[:5], m[5:9], m[9]
        alpha_rel = np.radians(inc - dip)
        tvd_rel = md * np.sin(alpha_rel)
        
        # Fronteras: El reservorio principal es la Capa 4
        z_int = np.cumsum(np.concatenate(([0], thick))) - np.sum(thick)/2
        
        response = np.full_like(md, res[0], dtype=float)
        for i in range(len(z_int)-1):
            weight = 0.5 * (1 + np.tanh((tvd_rel - z_int[i]) / 5.5))
            response = response * (1 - weight) + res[i+1] * weight
            
        return response * (1 + (ani - 1) * np.sin(alpha_rel)**2)

    def solve(self, mode, iters, obs, md, inc, dip):
        # Bounds definidos por litología:
        # [R_arcilla, R_arena, R_sello, R_reservorio, R_sello, Esp1, Esp2, Esp3, Esp4, Ani]
        bounds = [
            (0.5, 5),    # R1: Arcilla (Baja)
            (10, 50),    # R2: Arena
            (1, 8),      # R3: Sello intermedio
            (50, 500),   # R4: Reservorio Principal (Alta)
            (0.5, 10),   # R5: Sello Base
            (10, 30), (5, 20), (5, 20), (10, 30), # Espesores
            (1, 3)       # Anisotropía
        ]
        
        if mode == "Estocástico (Global)":
            res = differential_evolution(
                lambda m: np.sqrt(np.mean((obs - self.forward_model(m, md, inc, dip))**2)),
                bounds=bounds, maxiter=iters, popsize=10
            )
            return res.x, res.fun
        else:
            x0 = [2, 20, 4, 150, 2, 15, 10, 10, 20, 1.5]
            lb, ub = [b[0] for b in bounds], [b[1] for b in bounds]
            res = least_squares(
                lambda m: obs - self.forward_model(m, md, inc, dip),
                x0=x0, bounds=(lb, ub), max_nfev=iters
            )
            return res.x, np.mean(res.fun**2)
