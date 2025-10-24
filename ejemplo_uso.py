import numpy as np
from reto_opti.plantacion_imagen import generate_hex_lattice, plot_plantacion


def crear_aristas(puntos, distancia_maxima=3.5):
    """Crea una lista de aristas conectando nodos que están a distancia_maxima o menos.
    
    Args:
        puntos: numpy.ndarray de shape (n_points, 2) con coordenadas [x, y]
        distancia_maxima: distancia máxima para considerar dos nodos conectados
        
    Returns:
        aristas: lista de tuplas (i, j) donde i y j son índices de nodos conectados
    """
    n = len(puntos)
    aristas = []
    
    for i in range(n):
        for j in range(i + 1, n):  # j > i para evitar duplicados
            # Calcular distancia euclidiana
            distancia = np.linalg.norm(puntos[i] - puntos[j])
            
            if distancia <= distancia_maxima:
                aristas.append((i, j))
    
    return aristas


puntos = generate_hex_lattice(5000, 3.4, False)
aristas = crear_aristas(puntos, distancia_maxima=3.5)

print(f"Número de puntos: {len(puntos)}")
print(f"Número de aristas: {len(aristas)}")
print(f"Primeras 10 aristas: {aristas[:10]}")

# Exportar a JSON con la estructura solicitada
import json
output = {
    "nodes": [
        {"id": i, "x": float(p[0]), "y": float(p[1])}
        for i, p in enumerate(puntos)
    ],
    "edges": [list(e) for e in aristas]
}

with open("grafo_hexagonal.json", "w") as f:
    json.dump(output, f, indent=2)

print("Archivo 'grafo_hexagonal.json' generado correctamente.")

plot_plantacion(puntos, title="Grafo Hexagonal con Aristas")