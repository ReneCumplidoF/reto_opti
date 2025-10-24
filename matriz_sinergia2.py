import numpy as np

# 1. Datos numéricos transcritos de la imagen.
# La celda vacía en la esquina superior izquierda se representa como 'np.nan'
# (Not a Number), que es el estándar de NumPy para datos faltantes.

# Columnas: ['P.l.', 'Y.f.', 'O.st.', 'O.r.', 'O.e.', 'O.c.', 'A.sa.', 'A.sc.', 'A.st.', 'A.l.']
datos_matriz = np.array([
    [0.2, 0.2, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # Fila 'P. laevigata'
    [0.5,    0.2, 0.1, 0.1, 0.1, 0.1, 0.4, 0.4, 0.4, 0.4],  # Fila 'Y. filifera'
    [1.0,    0.1, 0.2, 0.2, 0.2, 0.2, 0.4, 0.4, 0.4, 0.4],  # Fila 'O. streptacantha'
    [1.0,    0.1, 0.2, 0.2, 0.2, 0.2, 0.4, 0.4, 0.4, 0.4],  # Fila 'O. robusta'
    [1.0,    0.1, 0.2, 0.2, 0.2, 0.2, 0.4, 0.4, 0.4, 0.4],  # Fila 'O. engelmannii'
    [1.0,    0.1, 0.2, 0.2, 0.2, 0.2, 0.4, 0.4, 0.4, 0.4],  # Fila 'O. cantabrigiens'
    [1.0,    0.4, 0.4, 0.4, 0.4, 0.4, 0.2, 0.4, 0.4, 0.4],  # Fila 'A. salmiana'
    [1.0,    0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.0, 0.2, 0.0],  # Fila 'A. scabra'
    [1.0,    0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.2, 0.0, 0.0],  # Fila 'A. striata'
    [1.0,    0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.0, 0.0, 0.0]   # Fila 'A. lechuguilla'
], dtype=float) # Es importante usar dtype=float para permitir 'np.nan'

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