"""
Genera gráficos de insights a partir de los outputs del GA+ACO y datos auxiliares.

Salidas (todas en output/figs/):
- ga_convergencia.png: Mejor costo del GA por generación
- aco_convergencia.png: Mejor costo del ACO por iteración
- cromosoma_barras.png: Conteos por especie vs baseline y bandas 90%-110%
- layout_especies.png: Dispersión de nodos coloreada por especie
- layout_calor_competencia.png: Calor de competencia por nodo
- matrices_especies.png: Matrices especie–especie (adyacencias y competencia)

Uso:
  python -m utils.graficos_insights
o
  python utils/graficos_insights.py
"""
from __future__ import annotations

from pathlib import Path
import json
from typing import Optional
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Asegurar que el root del proyecto esté en sys.path al ejecutar como script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import data_path, output_path


# --- Constantes y preparación ---
BASELINE = np.array([42, 196, 42, 42, 49, 38, 73, 64, 86, 26])
LB = np.floor(0.9 * BASELINE).astype(int)
UB = np.ceil(1.1 * BASELINE).astype(int)
SUMA_OBJ = 658

PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

# Directorio de figuras
FIGS_DIR: Path = output_path("figs")
FIGS_DIR.mkdir(parents=True, exist_ok=True)


def _exists(p: Path) -> bool:
    return p.exists() and p.is_file()


def plot_ga_convergencia() -> Optional[Path]:
    p_hist_ga = output_path("historial_costos_ga.npy")
    if not _exists(p_hist_ga):
        print(f"[skip] No existe {p_hist_ga}")
        return None
    hist = np.load(p_hist_ga)
    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(1, len(hist) + 1), hist, marker="o", linewidth=1.6)
    plt.xlabel("Generación")
    plt.ylabel("Mejor costo")
    plt.title("GA: Mejor costo por generación")
    plt.grid(True, alpha=0.3)
    out = FIGS_DIR / "ga_convergencia.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ok] {out}")
    return out


def plot_aco_convergencia() -> Optional[Path]:
    p_hist_aco = output_path("historial_costos_aco.npy")
    if not _exists(p_hist_aco):
        print(f"[skip] No existe {p_hist_aco}")
        return None
    hist = np.load(p_hist_aco)
    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(1, len(hist) + 1), hist, color="tab:orange", marker="o", linewidth=1.6)
    plt.xlabel("Iteración ACO")
    plt.ylabel("Mejor costo")
    plt.title("ACO: Mejor costo por iteración")
    plt.grid(True, alpha=0.3)
    out = FIGS_DIR / "aco_convergencia.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ok] {out}")
    return out


def plot_cromosoma_barras() -> Optional[Path]:
    p_crom = output_path("mejor_cromosoma_genetico.npy")
    if not _exists(p_crom):
        print(f"[skip] No existe {p_crom}")
        return None
    crom = np.load(p_crom)
    especies = np.arange(1, len(crom) + 1)
    plt.figure(figsize=(9, 4.8))
    # bandas
    plt.fill_between(especies, LB, UB, color="lightgray", alpha=0.5, label="Banda 90%-110%")
    # baseline
    plt.plot(especies, BASELINE, "--", color="black", label="Baseline")
    # barras
    plt.bar(especies, crom, color="tab:blue", alpha=0.85, label="Mejor cromosoma")
    plt.xticks(especies, [f"E{i}" for i in especies])
    plt.xlabel("Especie")
    plt.ylabel("Conteo")
    plt.title("Mejor cromosoma vs baseline y bandas")
    plt.legend()
    plt.grid(True, alpha=0.25)
    total = int(crom.sum())
    plt.text(0.02, 0.95, f"Suma={total} (objetivo {SUMA_OBJ})", transform=plt.gca().transAxes)
    out = FIGS_DIR / "cromosoma_barras.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ok] {out}")
    return out


def _cargar_grafo_y_asignacion():
    p_grafo = data_path("grafo_hexagonal.json")
    p_asig = output_path("mejor_asignacion_hormigas.npy")
    if not (_exists(p_grafo) and _exists(p_asig)):
        faltantes = []
        if not _exists(p_grafo):
            faltantes.append(str(p_grafo))
        if not _exists(p_asig):
            faltantes.append(str(p_asig))
        print("[skip] Faltan:", ", ".join(faltantes))
        return None, None, None
    with open(p_grafo, "r") as fh:
        grafo = json.load(fh)
    nodes = grafo["nodes"]
    edges = np.array(grafo["edges"], dtype=int)
    coords = np.array([[n["x"], n["y"]] for n in nodes], dtype=float)
    asig = np.load(p_asig)  # one-hot (n_nodos, n_tipos)
    tipos = np.argmax(asig, axis=1)
    return coords, edges, tipos


def plot_layout_especies() -> Optional[Path]:
    coords, edges, tipos = _cargar_grafo_y_asignacion()
    if coords is None:
        return None
    plt.figure(figsize=(7, 6))
    cmap = ListedColormap(PALETTE)
    sc = plt.scatter(coords[:, 0], coords[:, 1], c=tipos, s=10, cmap=cmap)
    plt.gca().set_aspect("equal")
    plt.title("Layout hexagonal por especie")
    plt.xlabel("x")
    plt.ylabel("y")
    cbar = plt.colorbar(sc, ticks=range(len(PALETTE)))
    cbar.ax.set_yticklabels([f"E{i+1}" for i in range(len(PALETTE))])
    out = FIGS_DIR / "layout_especies.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ok] {out}")
    return out


def plot_layout_calor_competencia() -> Optional[Path]:
    coords, edges, tipos = _cargar_grafo_y_asignacion()
    p_comp = data_path("matriz_competencia.npy")
    if coords is None or not _exists(p_comp):
        if not _exists(p_comp):
            print(f"[skip] No existe {p_comp}")
        return None
    comp = np.load(p_comp)
    n = len(tipos)
    calor = np.zeros(n, dtype=float)
    for i, j in edges:
        ti, tj = tipos[i], tipos[j]
        w = comp[ti, tj]
        calor[i] += w
        calor[j] += w
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(coords[:, 0], coords[:, 1], c=calor, s=12, cmap="inferno")
    plt.gca().set_aspect("equal")
    plt.title("Calor de competencia por nodo")
    plt.xlabel("x")
    plt.ylabel("y")
    cbar = plt.colorbar(sc)
    cbar.set_label("Competencia acumulada")
    out = FIGS_DIR / "layout_calor_competencia.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ok] {out}")
    return out


def plot_matrices_especies() -> Optional[Path]:
    coords, edges, tipos = _cargar_grafo_y_asignacion()
    p_comp = data_path("matriz_competencia.npy")
    if coords is None or not _exists(p_comp):
        if not _exists(p_comp):
            print(f"[skip] No existe {p_comp}")
        return None
    comp = np.load(p_comp)
    k = comp.shape[0]
    M_cnt = np.zeros((k, k), dtype=int)
    M_w = np.zeros((k, k), dtype=float)
    for i, j in edges:
        ti, tj = tipos[i], tipos[j]
        M_cnt[ti, tj] += 1
        M_cnt[tj, ti] += 1
        w = comp[ti, tj]
        M_w[ti, tj] += w
        M_w[tj, ti] += w
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    im0 = axes[0].imshow(M_cnt, cmap="Blues")
    axes[0].set_title("Adyacencias (conteo)")
    axes[0].set_xlabel("Especie j")
    axes[0].set_ylabel("Especie i")
    axes[0].set_xticks(range(k), [f"E{x+1}" for x in range(k)], rotation=45)
    axes[0].set_yticks(range(k), [f"E{x+1}" for x in range(k)])
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    im1 = axes[1].imshow(M_w, cmap="Oranges")
    axes[1].set_title("Competencia ponderada")
    axes[1].set_xlabel("Especie j")
    axes[1].set_ylabel("Especie i")
    axes[1].set_xticks(range(k), [f"E{x+1}" for x in range(k)], rotation=45)
    axes[1].set_yticks(range(k), [f"E{x+1}" for x in range(k)])
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = FIGS_DIR / "matrices_especies.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out}")
    return out


def main() -> None:
    print("Generando figuras en:", FIGS_DIR)
    generadas = []
    for fn in [
        plot_ga_convergencia,
        plot_aco_convergencia,
        plot_cromosoma_barras,
        plot_layout_especies,
        plot_layout_calor_competencia,
        plot_matrices_especies,
    ]:
        try:
            out = fn()
            if out is not None:
                generadas.append(out)
        except Exception as e:
            print(f"[error] {fn.__name__}: {e}")
    if len(generadas) == 0:
        print("No se generó ninguna figura (faltan archivos de entrada o hubo errores).")
    else:
        print("\nFiguras generadas:")
        for p in generadas:
            print("-", p)


if __name__ == "__main__":
    main()
