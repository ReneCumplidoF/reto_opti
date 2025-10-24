"""Generador de plantación en forma de malla triangular (hexagonal).

La función principal permite especificar la distancia entre plantas (spacing) y genera
una disposición regular donde cada punto forma triángulos equiláteros con sus vecinos.
Esta es la disposición clásica de "tres bolillos" en agricultura.

TIPOS DE DATOS:
- generate_hex_lattice() retorna: numpy.ndarray de shape (n_points, 2) con dtype float64
  Cada fila contiene [x, y] en las unidades especificadas (ej. metros)
- points_to_grid() retorna: numpy.ndarray de shape (height, width) con dtype uint8
  Matriz de ocupación donde 1 = celda ocupada, 0 = celda vacía
  
USO DESDE OTROS ARCHIVOS:
  from planteamiento import generate_hex_lattice, plot_plantacion
  coords = generate_hex_lattice(50, spacing=3.0)
  # coords es np.ndarray shape (50, 2) con coordenadas [x,y]
"""
from typing import Optional
import numpy as np
import math
import matplotlib.pyplot as plt


def generate_hex_lattice(n_points: int, spacing: float = 3.0,
                         center: bool = True) -> np.ndarray:
    """Genera hasta n_points coordenadas distribuidas en una malla triangular (hexagonal) con triángulos equiláteros.

    La malla se construye en filas con offset para formar triángulos equiláteros. Para una distancia entre plantas 'spacing' (p. ej. 3 m):
      - separación horizontal entre plantas contiguas en una fila: spacing
      - separación vertical entre filas: spacing * sqrt(3)/2
      - cada planta forma triángulos equiláteros con sus vecinos

    Args:
        n_points: número máximo de puntos a generar (la función devolverá exactamente n_points,
                  recortando la malla si es necesario).
        spacing: distancia entre plantas vecinas (lado del triángulo equilátero) en las mismas unidades (ej. metros).
        center: si True, traslada la malla para que sus coordenadas estén centradas alrededor del origen.

    Returns:
        points: numpy.ndarray de shape (n_points, 2) con dtype float64.
                Cada fila contiene [x, y] en las unidades especificadas.
                
    Example:
        >>> coords = generate_hex_lattice(10, spacing=3.0, center=False)
        >>> print(coords.shape)  # (10, 2)
        >>> print(coords.dtype)  # float64
        >>> x_coords = coords[:, 0]  # todas las coordenadas X
        >>> y_coords = coords[:, 1]  # todas las coordenadas Y
    """
    if n_points <= 0:
        return np.empty((0, 2))

    # estimar filas/columnas usando una malla aproximadamente cuadrada
    cols = int(math.ceil(math.sqrt(n_points)))
    rows = int(math.ceil(n_points / cols))

    dx = spacing
    dy = spacing * math.sqrt(3) / 2.0  # altura del triángulo equilátero

    coords = []
    for r in range(rows):
        # offset horizontal para filas impares para formar triángulos equiláteros
        x_offset = 0.5 * dx if (r % 2) == 1 else 0.0
        for c in range(cols):
            x = c * dx + x_offset
            y = r * dy
            coords.append((x, y))

    points = np.array(coords)
    # recortar al número pedido
    if points.shape[0] > n_points:
        points = points[:n_points]

    if center:
        # recentrar en torno al origen o cerca de la esquina (opcional)
        cx = points[:, 0].mean()
        cy = points[:, 1].mean()
        points[:, 0] -= cx
        points[:, 1] -= cy

    return points


def points_to_grid(points: np.ndarray, spacing: float = 3.0, pixels_per_unit: int = 10) -> np.ndarray:
    """Convierte coordenadas en unidades (ej. metros) a una matriz de ocupación.

    Args:
        points: (n,2) en unidades reales.
        spacing: usado solo para dimensionar la resolución relativa (informativo).
        pixels_per_unit: factor de escala para convertir unidades a pixeles/grilla.

    Returns:
        grid: matriz 2D uint8 donde 1 indica presencia de planta.
    """
    if points.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    # escalar y desplazar para indices positivos
    scaled = (points * pixels_per_unit).astype(int)
    min_x, min_y = scaled.min(axis=0)
    scaled[:, 0] -= min_x
    scaled[:, 1] -= min_y

    max_x, max_y = scaled.max(axis=0)
    grid = np.zeros((max_y + 1, max_x + 1), dtype=np.uint8)
    for x, y in scaled:
        # y como fila
        grid[y, x] = 1
    return grid


def plot_plantacion(points: np.ndarray, spacing: float = 3.0, show: bool = False, save_path: Optional[str] = None):
    """Grafica la plantación en malla triangular donde cada punto forma triángulos equiláteros.

    Dibuja puntos y las líneas que conectan vecinos para mostrar los triángulos equiláteros.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(points[:, 0], points[:, 1], s=50, color='tab:green', marker='*')
    ax.set_aspect('equal')
    ax.set_title(f'Plantación en malla triangular - triángulos equiláteros (spacing={spacing}m)')

    # Dibujar líneas de la malla triangular (conectar cada punto con vecinos cercanos)
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    # distancia de conexión: un poco mayor que spacing para capturar vecinos directos
    thresh = spacing * 1.01
    pairs = tree.query_pairs(r=thresh)
    for i, j in pairs:
        x0, y0 = points[i]
        x1, y1 = points[j]
        ax.plot([x0, x1], [y0, y1], color='lightblue', linewidth=0.8, alpha=0.7)

    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x (metros)')
    ax.set_ylabel('y (metros)')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    # demo: generar n puntos dispuestos con distancia entre plantas especificada
    n = 600  # aproximadamente 5x7 en disposición triangular
    spacing = 3.4  # distancia entre plantas (metros)
    pts = generate_hex_lattice(500, spacing=spacing, center=False)
    plot_plantacion(pts, spacing=spacing, save_path='plantacion_triangular.png')
    print('Imagen guardada en plantacion_triangular.png')
    print(f'Generados {len(pts)} puntos en malla triangular (triángulos equiláteros) con spacing={spacing}m')
