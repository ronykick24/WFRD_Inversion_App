import numpy as np
from scipy.optimize import differential_evolution
# IMPORTANTE: El nombre coincide con tu archivo en GitHub
from physics_engine import calculate_3d_horns

def run_proactive_inversion(measured_res, inc_val, layers):
    """
    Algoritmo Estocástico Proactivo:
    Optimiza el Shift y el DIP comparando el dato real vs el modelo físico.
    """
    def objective(params):
        s_test, d_test = params
        # Capa 2 es el Reservorio Target en nuestro modelo de app.py
        rh, rv = layers[2]['rh'], layers[2]['rv']
        
        # Llamamos a la física para generar el modelo sintético
        synth = calculate_3d_horns(rh, rv, inc_val, d_test, s_test)
        
        # Error cuadrático logarítmico (Misfit)
        return (np.log10(measured_res) - np.log10(synth))**2

    # Límites de búsqueda: Shift +/- 60 pies, DIP +/- 15 grados
    result = differential_evolution(objective, bounds=[(-60, 60), (-15, 15)])
    return result.x # Retorna [Mejor_Shift, Mejor_DIP]
