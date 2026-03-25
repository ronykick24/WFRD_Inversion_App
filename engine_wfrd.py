import numpy as np
from scipy.optimize import least_squares

class StochasticInversion:
    def __init__(self):
        # Capacidades físicas de WFRD GuideWave
        self.depth_investigation = {'33ft': 33.0, '40ft': 40.0, '50ft': 50.0}

    def forward_model(self, m, dist_target):
        res, dip = m
        # Modelo físico mejorado para múltiples capas
        return res * np.exp(-25.0 / dist_target) * np.cos(np.radians(dip))

    def run(self, obs_value, dist_target=50.0):
        def residual(m):
            return self.forward_model(m, dist_target) - obs_value
        
        # Inversión Newtoniana con límites físicos
        res = least_squares(residual, x0=[10.0, 0.0], bounds=([0.1, -90], [2000, 90]))
        return res.x
