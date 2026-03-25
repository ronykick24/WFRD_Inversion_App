import numpy as np
from scipy.optimize import differential_evolution

class StochasticInversion5L:
    def __init__(self):
        self.reach = 50.0

    def forward_model(self, m, md, inc, dip):
        # m = [R1, R2, R3, R4, R5, espesor1, esp2, esp3, esp4, ani_ratio]
        res = m[:5]
        thick = m[5:9]
        lam = m[9] # Anisotropía
        
        # Geometría: Ángulo relativo entre pozo y formación
        rel_angle = np.radians(inc - dip)
        
        # TVD relativo (proyección vertical del pozo respecto a las capas)
        tvd_pos = md * np.cos(rel_angle)
        
        # Definición de interfaces de las 5 capas
        interfaces = np.cumsum(thick) - np.sum(thick)/2
        
        # Respuesta con corrección de anisotropía y transiciones suaves
        # A 85°, la resistividad aparente se ve afectada por lam (Rv/Rh)
        response = res[0]
        for i in range(len(interfaces)):
            # Función de transición sigmoide para simular la física de la herramienta
            transition = 0.5 * (1 + np.tanh((tvd_pos - interfaces[i]) / 5.0))
            response += (res[i+1] - res[i]) * transition
            
        return response * (1 + (lam - 1) * np.sin(rel_angle)**2)

    def run_inversion(self, obs_data, md_array, inc, dip_guess):
        # Rangos: Res [0.1-1000], Espesores [2-25ft], Anisotropia [1-4]
        bounds = [(0.1, 500)]*5 + [(5, 30)]*4 + [(1, 4)]
        
        # Algoritmo Estocástico de Evolución Diferencial (Global)
        result = differential_evolution(
            lambda m: np.sqrt(np.mean((obs_data - self.forward_model(m, md_array, inc, dip_guess))**2)),
            bounds=bounds, 
            popsize=10,
            tol=0.01
        )
        return result.x, result.fun
