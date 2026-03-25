import numpy as np
from scipy.optimize import differential_evolution

class StochasticInversion5L:
    def __init__(self):
        self.reach = 50.0

    def forward_model(self, m, md, inc, dip):
        # m = [R1, R2, R3, R4, R5, h1, h2, h3, h4]
        res = m[:5]
        thick = m[5:]
        angle_rel = np.radians(inc - dip)
        
        # Cálculo de TVD relativo al pozo
        tvd_pos = md * np.cos(np.radians(inc))
        
        # Posición de las interfaces
        interfaces = np.cumsum(thick) - np.sum(thick)/2
        
        # Contribución multicapa con suavizado de transición
        val = res[0]
        for i in range(len(interfaces)):
            val += (res[i+1] - res[i]) * 0.5 * (1 + np.tanh((tvd_pos - interfaces[i]) / 2))
        return val

    def calculate_misfit(self, obs, pred):
        return np.sqrt(np.mean((obs - pred)**2))

    def run_inversion(self, obs_data, md, inc, dip_guess):
        # Usamos Evolución Diferencial (Estocástico) para encontrar el mínimo global
        bounds = [(0.1, 1000)]*5 + [(2, 20)]*4 # Res [0.1-1000], Espesores [2-20ft]
        
        result = differential_evolution(
            lambda m: self.calculate_misfit(obs_data, self.forward_model(m, md, inc, dip_guess)),
            bounds=bounds, tol=0.01
        )
        return result.x, result.fun # Retorna parámetros y el Misfit
