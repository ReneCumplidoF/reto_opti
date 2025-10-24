import numpy as np

# Generar matriz de sinergia aleatoria para 10 tipos de plantas
n_tipos = 10
np.random.seed(42)  # Para reproducibilidad

# Valores entre 0 y 1, distribución normal truncada
from scipy.stats import truncnorm
a, b = (0 - 0.5) / 0.15, (1 - 0.5) / 0.15  # media=0.5, std=0.15, truncado a [0,1]
sinergia = truncnorm.rvs(a, b, loc=0.5, scale=0.15, size=(n_tipos, n_tipos))

# Hacer la matriz simétrica
sinergia = (sinergia + sinergia.T) / 2

# Guardar la matriz de sinergia
np.save("matriz_sinergia.npy", sinergia)
print("Matriz de sinergia generada y guardada como 'matriz_sinergia.npy'.")
print("Sinergia (primeras filas):\n", sinergia[:3])
