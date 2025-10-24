import numpy as np

# 1. Datos numéricos transcritos de la imagen.
# La celda vacía en la esquina superior izquierda se representa como 'np.nan'
# (Not a Number), que es el estándar de NumPy para datos faltantes.

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


# 2. Nombre del archivo de salida
nombre_archivo = 'matriz_de_similitud.npy'

# 3. Guardar el array en el archivo .npy
np.save(nombre_archivo, datos_matriz)

print(f"Array de NumPy guardado exitosamente como '{nombre_archivo}'")
print(f"Dimensiones del array: {datos_matriz.shape}")

# --- Opcional: Cómo cargar y verificar el archivo ---
print("\n--- Verificación ---")
# Cargar el archivo .npy
datos_cargados = np.load(nombre_archivo)

print(f"Datos cargados desde '{nombre_archivo}':")
print(datos_cargados)