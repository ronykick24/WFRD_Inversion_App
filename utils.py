def get_wfrd_palette():
    """Paleta técnica para predecir contacto agua-aceite."""
    return [
        [0.0, "#000044"],   # Agua Profunda
        [0.2, "#0055FF"],   # Acuífero
        [0.5, "#FFFF00"],   # Oil Leg (Amarillo Brillante)
        [0.8, "#8B4513"],   # Transición
        [1.0, "#331100"]    # Sello / Shale
    ]
