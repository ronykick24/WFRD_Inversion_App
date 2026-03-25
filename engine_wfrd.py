import numpy as np
from scipy.optimize import differential_evolution

class WFRD_Simulator:
    def __init__(self):
        self.reach = 50.0  # Alcance máximo de la herramienta

    def forward_model_2D(self, m, md, inc, dip):
        """
        m = [R_capas(5), Espesores(4), Anisotropia, Posicion_Contacto]
        Simula la transición física de la onda a través de interfaces.
        """
        res_layers = m[:5]
        thicknesses = m[5:9]
        ani_ratio = m[9]
        contact_z = m[10]

        # Geometría: Ángulo de ataque relativo
        alpha = np.radians(inc - dip)
        
        # TVD relativo proyectado
        tvd_rel = (md * np.cos(alpha)) - contact_z
        
        # Definición de las interfaces de las 5 capas
        z_interfaces = np.cumsum(np.insert(thicknesses, 0, 0)) - np.sum(thicknesses)/2
        
        # Simulación de la respuesta (Mezcla de capas por sensibilidad volumétrica)
        response = res_layers[0]
        for i in range(len(z_interfaces)-1):
            # Función sigmoide para representar la resolución vertical de la herramienta
            weight = 0.5 * (1 + np.tanh((tvd_rel - z_interfaces[i]) / 4.0))
            response = response * (1 - weight) + res_layers[i+1] * weight
            
        return response

    def solve(self, obs_data, md_array, inc):
        # Buscamos: 5 Resistividades, 4 Espesores, 1 Anisotropía, 1 Dip
        bounds = [(0.1, 1000)]*5 + [(2, 25)]*4 + [(1, 5)] + [(-10, 10)]
        
        result = differential_evolution(
            lambda m: np.mean((obs_data - self.forward_model_2D(m[:-1], md_array, inc, m[-1]))**2),
            bounds=bounds, popsize=15
        )
        return result.x # Retorna el modelo de la formación y el Dip predicho
