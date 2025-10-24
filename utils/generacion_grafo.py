import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.plantacion_imagen import generate_hex_lattice, plot_plantacion
from utils.paths import data_path, output_path


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


puntos = generate_hex_lattice(658, 3.4, False)
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

with open(data_path("grafo_hexagonal.json"), "w") as f:
    json.dump(output, f, indent=2)

print(f"Archivo guardado en '{data_path('grafo_hexagonal.json')}'")

plot_plantacion(puntos, spacing=3.4, show=False, save_path=output_path("grafo_hexagonal.png"))
print(f"Gráfico guardado en '{output_path('grafo_hexagonal.png')}'")