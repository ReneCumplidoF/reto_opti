import numpy as np
import json

# Cargar el grafo desde el archivo JSON
grafo_path = "grafo_hexagonal.json"
with open(grafo_path, "r") as f:
    grafo = json.load(f)

n_nodos = len(grafo["nodes"])
n_tipos = 10

# Crear una matriz dummy de asignación de plantas
# Cada fila es un nodo, cada columna un tipo de planta
# La suma total de plantas debe ser 5000 (una por nodo)

# Distribución aleatoria de tipos de plantas
plantas_por_tipo = np.random.multinomial(n_nodos, [1/n_tipos]*n_tipos)

# Crear la matriz de asignación (n_nodos x n_tipos)
matriz_plantas = np.zeros((n_nodos, n_tipos), dtype=int)

# Asignar los tipos de plantas a los nodos
tipo_actual = 0
contador = 0
for i in range(n_nodos):
    while plantas_por_tipo[tipo_actual] == 0:
        tipo_actual += 1
    matriz_plantas[i, tipo_actual] = 1
    plantas_por_tipo[tipo_actual] -= 1

# Guardar la matriz dummy en un archivo para su uso posterior
np.save("matriz_dummy_plantas.npy", matriz_plantas)

print("Matriz dummy de plantas generada y guardada como 'matriz_dummy_plantas.npy'.")
print("Shape:", matriz_plantas.shape)
print("Suma total:", matriz_plantas.sum())
print("Plantas por tipo:", matriz_plantas.sum(axis=0))
