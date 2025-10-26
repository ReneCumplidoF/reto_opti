"""
Visualiza las plantas iniciales (1 hectárea fija) en el layout hexagonal.

Lee: 
  - datos/hectarea.json (asignación inicial)
  - datos/grafo_hexagonal.json (coordenadas)
Genera: output/figs/plantas_iniciales.png

Uso:
  python -m utils.grafico_plantas_iniciales
o
  python utils/grafico_plantas_iniciales.py
"""
from __future__ import annotations

from pathlib import Path
import sys
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# Asegurar imports locales
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import data_path, output_path

PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

NOMBRES_ESPECIES = [
    "Agave lechuguilla",
    "Agave salmiana",
    "Agave scabra",
    "Agave striata",
    "Opuntia cantabrigiensis",
    "Opuntia engelmannii",
    "Opuntia robusta",
    "Opuntia streptacantha",
    "Prosopis laevigata",
    "Yucca filifera"
]


def plot_plantas_iniciales():
    # Cargar grafo y coordenadas
    p_grafo = data_path("grafo_hexagonal.json")
    p_hectarea = data_path("hectarea.json")
    
    if not (p_grafo.exists() and p_hectarea.exists()):
        print(f"❌ Faltan archivos: {p_grafo} o {p_hectarea}")
        return None
    
    with open(p_grafo, 'r') as f:
        grafo = json.load(f)
    
    with open(p_hectarea, 'r') as f:
        data_hect = json.load(f)
    
    nodes = grafo['nodes']
    coords = np.array([[n['x'], n['y']] for n in nodes], dtype=float)
    
    asig_inicial = data_hect.get('asignacion_indices', data_hect.get('asignacion'))
    tipos = np.array(asig_inicial, dtype=int)
    conteos = data_hect['conteos_por_especie']
    
    # Separar nodos con plantas vs vacíos
    plantas_mask = tipos >= 0
    vacios_mask = tipos == -1
    
    n_plantas = plantas_mask.sum()
    n_vacios = vacios_mask.sum()
    
    print(f"Plantas iniciales: {n_plantas}")
    print(f"Nodos vacíos: {n_vacios}")
    print(f"Total: {len(tipos)}")
    
    # Crear figura de un solo panel (como el ejemplo)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Dibujar nodos vacíos primero (en gris claro)
    if n_vacios > 0:
        ax.scatter(coords[vacios_mask, 0], coords[vacios_mask, 1], 
                   c='#D3D3D3', s=20, alpha=0.5, marker='o')
    
    # Dibujar plantas por especie
    if n_plantas > 0:
        for i in range(10):
            mask_especie = tipos == i
            if mask_especie.sum() > 0:
                ax.scatter(coords[mask_especie, 0], coords[mask_especie, 1],
                           c=PALETTE[i], s=60, marker='*', 
                           edgecolors='black', linewidths=0.3, alpha=0.95)
    
    ax.set_aspect('equal')
    ax.set_title(f'Layout Hexagonal - Plantas Iniciales (1 Ha)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    
    # Crear leyenda colorbar manual en el lado derecho
    from matplotlib.patches import Rectangle
    
    # Posición de la barra de color (a la derecha del gráfico)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
    
    # Crear la barra de color manualmente
    for i in range(10):
        rect = Rectangle((0, i), 1, 1, facecolor=PALETTE[i], edgecolor='black', linewidth=0.5)
        cbar_ax.add_patch(rect)
    
    cbar_ax.set_xlim(0, 1)
    cbar_ax.set_ylim(0, 10)
    cbar_ax.set_yticks([i + 0.5 for i in range(10)])
    cbar_ax.set_yticklabels([f'E{i+1}' for i in range(10)], fontsize=10)
    cbar_ax.set_xticks([])
    cbar_ax.spines['top'].set_visible(True)
    cbar_ax.spines['right'].set_visible(True)
    cbar_ax.spines['bottom'].set_visible(True)
    cbar_ax.spines['left'].set_visible(True)
    
    # Agregar info adicional
    info_text = f'Plantas fijas: {n_plantas}\nEspacios vacíos: {n_vacios}\nTotal nodos: {len(tipos)}'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.tight_layout()
    
    out = output_path("figs") / "plantas_iniciales.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Gráfico generado: {out}")
    
    # Imprimir tabla resumen
    print("\nTabla de especies:")
    print(f"{'ID':<5} {'Nombre':<30} {'Cantidad':<10} {'%'}")
    print("-"*60)
    for i, (nombre, cant) in enumerate(zip(NOMBRES_ESPECIES, conteos.values())):
        print(f"E{i+1:<4} {nombre:<30} {cant:<10} {100*cant/n_plantas:.1f}%")
    
    return out


def main():
    plot_plantas_iniciales()


if __name__ == "__main__":
    main()
