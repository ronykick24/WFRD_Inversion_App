import pandas as pd

def get_owc_palette():
    """Paleta de Alto Relieve para Contacto Agua-Aceite."""
    return [
        [0.0, "#000032"], [0.3, "#00C8FF"], # Agua
        [0.6, "#FFD700"], [1.0, "#4b2c20"]  # Aceite / Sello
    ]

def save_geosteering_report(data):
    """Convierte diccionario de datos en DataFrame."""
    return pd.DataFrame([data])
