import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import data_path

# 1. Datos numéricos de la matriz de competencia entre especies.
# Valores altos indican alta competencia (queremos MINIMIZAR).
# La diagonal es 1.0 (una planta compite al máximo consigo misma/su tipo)

datos_matriz = np.array([
    [1.0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.3, 0.8],  # Agave lechuguilla
    [0.9, 1.0, 0.8, 0.8, 0.7, 0.7, 0.7, 0.7, 0.2, 0.8],  # Agave salmiana
    [0.9, 0.8, 1.0, 0.8, 0.7, 0.7, 0.7, 0.7, 0.2, 0.8],  # Agave scabra
    [0.9, 0.8, 0.8, 1.0, 0.7, 0.7, 0.7, 0.7, 0.2, 0.8],  # Agave striata
    [0.9, 0.7, 0.7, 0.7, 1.0, 0.8, 0.8, 0.8, 0.2, 0.8],  # Opuntia cantabrigiensis
    [0.9, 0.7, 0.7, 0.7, 0.8, 1.0, 0.8, 0.8, 0.2, 0.8],  # Opuntia engelmannii
    [0.9, 0.7, 0.7, 0.7, 0.8, 0.8, 1.0, 0.8, 0.2, 0.8],  # Opuntia robusta
    [0.9, 0.7, 0.7, 0.7, 0.8, 0.8, 0.8, 1.0, 0.2, 0.8],  # Opuntia streptacantha
    [0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 1.0, 0.5],  # Prosopis laevigata
    [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.5, 1.0]   # Yucca filifera
], dtype=float)


# 2. Nombre del archivo de salida (ahora guardamos en datos/)
nombre_archivo = data_path('matriz_competencia.npy')

# 3. Guardar el array en el archivo .npy
np.save(nombre_archivo, datos_matriz)

print(f"Matriz de competencia guardada en '{nombre_archivo}'")
print(f"Dimensiones del array: {datos_matriz.shape}")

# --- Opcional: Cómo cargar y verificar el archivo ---
print("\n--- Verificación ---")
# Cargar el archivo .npy
datos_cargados = np.load(nombre_archivo)

print(f"Datos cargados desde '{nombre_archivo}':")
print(datos_cargados)
print("\nNOTA: Esta matriz representa COMPETENCIA (valores altos = mala compatibilidad)")
print("El algoritmo debe MINIMIZAR la suma de competencias entre vecinos.")