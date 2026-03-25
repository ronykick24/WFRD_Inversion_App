import numpy as np
from scipy.optimize import least_squares

class StochasticInversion:
    def __init__(self):
        self.range_50 = 50.0

    def forward_model_anisotropic(self, params, inc_deg):
        # m = [Rh, Rv, Distancia_Capa]
        rh, rv, dist = params
        theta = np.radians(inc_deg)
        
        # Coeficiente de Anisotropía Lambda
        lam = np.sqrt(rv / rh)
        
        # Respuesta corregida por inclinación (85 deg)
        # A alto ángulo, la componente vertical domina la fase
        res_eff = rh / np.sqrt(np.cos(theta)**2 + (1/lam**2) * np.sin(theta)**2)
        
        # Atenuación geométrica hacia la capa a 50ft
        return res_eff * np.exp(-dist / self.range_50)

    def run(self, obs_value, inc_actual):
        def objective(p):
            # p = [Rh, Rv, Distancia]
            return self.forward_model_anisotropic(p, inc_actual) - obs_value
        
        # Inversión con límites: Rh [0.1-2000], Rv [0.1-4000], Dist [0-50]
        res = least_squares(objective, x0=[10, 20, 10], 
                            bounds=([0.1, 0.1, 0], [2000, 4000, 50]))
        return res.x
