import numpy as np
from scipy.optimize import least_squares

class InversionEngine:
    def __init__(self):
        self.max_range = 50.0 # Capacidad física WFRD

    def forward_model(self, m, dist):
        """ Modelo matemático de propagación para 33ft y 50ft """
        res_layer, dip = m
        # Ecuación de caída electromagnética simplificada
        return res_layer * np.exp(-dist / self.max_range) * np.cos(np.radians(dip))

    def run_stochastic_inversion(self, obs_data, initial_guess=[10, 0]):
        """ Algoritmo Gauss-Newton con paso estocástico """
        def residual(m):
            return self.forward_model(m, 25.0) - obs_data
        
        # Inversión rápida Newton-Marquardt
        res = least_squares(residual, initial_guess, bounds=((0.1, -90), (2000, 90)))
        return res.x # Retorna [Resistividad, Dip]
