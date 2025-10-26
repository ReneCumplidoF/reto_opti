"""
Visualiza la evolución de los cromosomas (cantidades por especie) a través de las generaciones del GA.

Lee: output/historial_cromosomas.npy (shape: [n_gens, 10])
Genera: output/figs/evolucion_cromosomas.png

Uso:
  python -m utils.grafico_evolucion_cromosomas
o
  python utils/grafico_evolucion_cromosomas.py
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Asegurar imports locales
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import output_path

# Constantes
BASELINE = np.array([42, 196, 42, 42, 49, 38, 73, 64, 86, 26])
LB = np.floor(0.9 * BASELINE).astype(int)
UB = np.ceil(1.1 * BASELINE).astype(int)

PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

NOMBRES_ESPECIES = [f"E{i+1}" for i in range(10)]


def plot_evolucion_cromosomas():
    p_hist = output_path("historial_cromosomas.npy")
    if not (p_hist.exists() and p_hist.is_file()):
        print(f"❌ No existe {p_hist}")
        print("   Ejecuta el GA paralelo primero para generar el historial.")
        return None

    hist = np.load(p_hist)  # shape: (n_gens, 10)
    n_gens, n_especies = hist.shape

    print(f"Historial de cromosomas cargado: {n_gens} generaciones × {n_especies} especies")

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 2, 1], hspace=0.35, wspace=0.3)

    # --- Panel 1: Evolución por especie (líneas) ---
    ax1 = fig.add_subplot(gs[0, :])
    generaciones = np.arange(1, n_gens + 1)
    
    for i in range(n_especies):
        ax1.plot(generaciones, hist[:, i], label=NOMBRES_ESPECIES[i], 
                color=PALETTE[i], linewidth=1.8, marker='o', markersize=3, alpha=0.85)
    
    # Líneas de baseline
    for i in range(n_especies):
        ax1.axhline(BASELINE[i], color=PALETTE[i], linestyle='--', linewidth=0.8, alpha=0.3)
    
    ax1.set_xlabel("Generación", fontsize=11)
    ax1.set_ylabel("Cantidad de plantas", fontsize=11)
    ax1.set_title("Evolución de las cantidades por especie a través del GA", fontsize=13, fontweight='bold')
    ax1.legend(ncol=5, fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.25)

    # --- Panel 2: Heatmap de evolución ---
    ax2 = fig.add_subplot(gs[1, :])
    im = ax2.imshow(hist.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax2.set_xlabel("Generación", fontsize=11)
    ax2.set_ylabel("Especie", fontsize=11)
    ax2.set_title("Mapa de calor: cantidad por (especie, generación)", fontsize=13, fontweight='bold')
    ax2.set_yticks(range(n_especies), NOMBRES_ESPECIES)
    
    # Marcar cada X generaciones en el eje X
    step = max(1, n_gens // 20)
    xticks = list(range(0, n_gens, step)) + [n_gens - 1]
    ax2.set_xticks(xticks, [str(g + 1) for g in xticks])
    
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Cantidad", fontsize=10)

    # --- Panel 3: Diferencias finales vs baseline ---
    ax3 = fig.add_subplot(gs[2, 0])
    mejor_final = hist[-1]
    deltas = mejor_final - BASELINE
    colores_barras = [PALETTE[i] for i in range(n_especies)]
    bars = ax3.bar(NOMBRES_ESPECIES, deltas, color=colores_barras, alpha=0.8, edgecolor='black', linewidth=0.8)
    ax3.axhline(0, color='black', linewidth=1.2, linestyle='-')
    ax3.set_ylabel("Δ vs baseline", fontsize=10)
    ax3.set_title("Cambio final respecto al baseline", fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.25, axis='y')
    
    # Anotar valores
    for bar, delta in zip(bars, deltas):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(delta):+d}', ha='center', va='bottom' if delta >= 0 else 'top', fontsize=8)

    # --- Panel 4: Distancia a bandas ---
    ax4 = fig.add_subplot(gs[2, 1])
    dist_lb = np.maximum(0, LB - mejor_final)  # si está por debajo de LB
    dist_ub = np.maximum(0, mejor_final - UB)  # si está por encima de UB
    violaciones = dist_lb + dist_ub
    
    bars2 = ax4.bar(NOMBRES_ESPECIES, violaciones, color='crimson', alpha=0.8, edgecolor='black', linewidth=0.8)
    ax4.set_ylabel("Distancia fuera de banda", fontsize=10)
    ax4.set_title("Violaciones de restricciones 90%-110%", fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.25, axis='y')
    
    n_viols = int((violaciones > 0).sum())
    ax4.text(0.5, 0.95, f'{n_viols} especie(s) fuera de rango',
             transform=ax4.transAxes, ha='center', va='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    out = output_path("figs") / "evolucion_cromosomas.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Gráfico generado: {out}")
    return out


def main():
    plot_evolucion_cromosomas()


if __name__ == "__main__":
    main()
