#!/usr/bin/env python3
import json
import csv
import numpy as np
from typing import List, Dict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import data_path, output_path

GRAFO_PATH = data_path("grafo_hexagonal.json")
CSV_PATH = data_path("info_actual.csv")
OUTPUT_PATH = output_path("acomodos.json")
N_ACOMODOS = 10


def cargar_grafo_n_nodos(path: str) -> int:
    with open(path, "r") as f:
        grafo = json.load(f)
    return len(grafo["nodes"])  # cantidad de posiciones en la malla


def cargar_densidades_desde_csv(csv_path: str):
    especies: List[str] = []
    densidades: List[np.ndarray] = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        filas = list(reader)

    # Identificar la fila de áreas
    area_row = None
    for row in filas:
        if row and "Área" in row[0]:
            area_row = row
            break
    if area_row is None:
        raise ValueError("No se encontró la fila de 'Área del polígono (Ha)' en el CSV")

    areas = np.array([float(x) for x in area_row[1:]], dtype=float)
    if not np.all(areas > 0):
        raise ValueError("Se encontraron áreas no positivas en el CSV")

    for row in filas:
        nombre = row[0]
        if "Área" in nombre:
            continue
        # Conteos observados en los polígonos
        conteos = np.array([float(x) for x in row[1:]], dtype=float)
        # Densidades por polígono para la especie
        dens = conteos / areas
        especies.append(nombre)
        densidades.append(dens)

    return especies, densidades


def ajustar_a_total(counts_float: np.ndarray, total: int) -> np.ndarray:
    base = np.floor(counts_float).astype(int)
    diff = total - base.sum()
    if diff == 0:
        return base
    # distribuir según parte fraccional
    frac = counts_float - np.floor(counts_float)
    order = np.argsort(-frac)  # descendente
    if diff > 0:
        base[order[:diff]] += 1
    else:
        # si hay exceso (raro), restamos en los más pequeños fraccionalmente
        order_asc = np.argsort(frac)
        base[order_asc[:abs(diff)]] = np.maximum(0, base[order_asc[:abs(diff)]] - 1)
    return base


def generar_acomodos(n_acomodos: int = N_ACOMODOS) -> Dict:
    n_nodos = cargar_grafo_n_nodos(GRAFO_PATH)
    especies, densidades = cargar_densidades_desde_csv(CSV_PATH)
    n_tipos = len(especies)

    # Densidad promedio por especie (para determinar el área equivalente)
    mean_dens = np.array([d.mean() for d in densidades], dtype=float)
    suma_mean = float(mean_dens.sum())
    if suma_mean <= 0:
        raise ValueError("La suma de densidades medias es 0; verifique el CSV")

    # Elegimos un área (Ha) tal que el total esperado ≈ n_nodos
    area_equivalente = n_nodos / suma_mean

    acomodos = []
    rng = np.random.default_rng()

    for k in range(n_acomodos):
        # Bootstrap: tomar una densidad aleatoria (con reemplazo) por especie
        dens_sample = np.array([rng.choice(d) for d in densidades], dtype=float)
        # Conteos esperados para el área equivalente
        counts_float = dens_sample * area_equivalente
        counts = ajustar_a_total(counts_float, n_nodos)

        # Construir una asignación aleatoria de nodos a especies respetando los conteos
        nodos = np.arange(n_nodos)
        rng.shuffle(nodos)
        asignacion_idx = np.empty(n_nodos, dtype=int)
        start = 0
        for idx_especie, c in enumerate(counts):
            end = start + int(c)
            if end > n_nodos:
                end = n_nodos
            asignacion_idx[nodos[start:end]] = idx_especie
            start = end
        # Si por algún motivo no se completó (no debería), rellenar con la última especie
        if start < n_nodos:
            asignacion_idx[nodos[start:]] = n_tipos - 1

        conteos_dict = {especies[i]: int(counts[i]) for i in range(n_tipos)}

        acomodos.append({
            "id": k + 1,
            "semilla": int(rng.integers(0, 2**32 - 1)),
            "conteos": conteos_dict,
            "asignacion": asignacion_idx.tolist()
        })

    salida = {
        "metadata": {
            "n_nodos": n_nodos,
            "n_tipos": n_tipos,
            "especies": especies,
            "area_equivalente_ha": area_equivalente,
            "fuente": "info_actual.csv",
            "metodo": "bootstrap_densidades_por_especie",
        },
        "acomodos": acomodos
    }
    return salida


def main():
    resultado = generar_acomodos(N_ACOMODOS)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)
    print(f"✓ Acomodos guardados en {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
