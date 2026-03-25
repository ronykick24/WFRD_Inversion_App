import numpy as np

def get_vertical_profile(shift, layers, z_range=100):
    """Genera un perfil de resistividad vertical basado en el modelo de capas."""
    z_coords = np.linspace(shift - z_range/2, shift + z_range/2, 100)
    res_profile = []
    
    for z in z_coords:
        # Lógica para determinar en qué capa cae cada punto Z
        cumulative_z = shift
        current_res = layers[-1]['rh']
        for ly in layers:
            if z <= cumulative_z and z > (cumulative_z - ly['tst']):
                current_res = ly['rh']
                break
            cumulative_z -= ly['tst']
        res_profile.append(current_res)
    return z_coords, np.array(res_profile)
