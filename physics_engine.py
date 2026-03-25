# physics_engine.py - Actualizado con sensibilidad de interfaz (ScienceDirect)

def calculate_ahta_sensitivity(dist, rel_angle, mode='look-around'):
    """
    Ajusta la respuesta según la sensibilidad del ángulo de interfaz 
    extraída del artículo de ScienceDirect 2024.
    """
    # El artículo indica que el modo Look-ahead es más sensible a 
    # contrastes de alta resistividad a mayor distancia.
    if mode == 'look-ahead':
        sensitivity = np.exp(-abs(dist) / 15.0) * np.cos(np.radians(rel_angle))
    else: # look-around
        sensitivity = np.exp(-abs(dist) / 5.5) * np.sin(np.radians(rel_angle))
    
    return sensitivity
