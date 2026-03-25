import numpy as np
from scipy.optimize import least_squares

class StochasticInversion:
    def __init__(self):
        self.max_reach = 50.0 # Alcance físico WFRD

    def simulate_layer_response(self, m, md_points, inc_deg):
        """
        m = [Rh_capa1, Rh_capa2, espesor_capa, tvd_contacto, anisotropy_ratio]
        """
        rh1, rh2, thickness, contact_tvd, alan = m
        rv1 = rh1 * (alan**2)
        
        # Proyectar MD a TVD para ver el cruce real a 85°
        tvd_points = md_points * np.cos(np.radians(inc_deg))
        
        # Calcular respuesta basada en la distancia al contacto
        dist_to_contact = tvd_points - contact_tvd
        
        # Modelo de transición suave (Física de la inducción)
        # La herramienta empieza a "ver" la capa 50ft antes de tocarla
        response = rh1 + (rh2 - rh1) * 0.5 * (1 + np.tanh(dist_to_contact / (self.max_reach / 2)))
        
        return response

    def run_stochastic_inversion(self, obs_data, md_points, inc_actual):
        # Buscamos el mejor ajuste para espesor y resistividades
        def objective(p):
            return self.simulate_layer_response(p, md_points, inc_actual) - obs_data
        
        # p0 = [Rh1, Rh2, Espesor, TVD_inicial, Anisotropia]
        p0 = [2.0, 50.0, 15.0, 10.0, 1.5]
        res = least_squares(objective, p0, bounds=([0.1, 0.1, 1, -100, 1], [1000, 1000, 200, 100, 5]))
        return res.x
