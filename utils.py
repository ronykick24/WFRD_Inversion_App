import pandas as pd

def get_owc_palette():
    """Paleta de Alto Relieve para Contacto Agua-Aceite."""
    return [
        [0.0, "#000032"], [0.3, "#00C8FF"], # Acuífero
        [0.6, "#FFD700"], [1.0, "#4b2c20"]  # Reservorio / Sello
    ]

def save_geosteering_report(data):
    """Crea un DataFrame listo para descargar."""
    return pd.DataFrame([data])
