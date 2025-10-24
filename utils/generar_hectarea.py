#!/usr/bin/env python3
import csv
import json
import numpy as np
from typing import List, Dict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import data_path, output_path, root_path

CSV_PATH = data_path("info_actual.csv")
OUTPUT_JSON = data_path("hectarea.json")
OUTPUT_TXT = root_path("hectarea.txt")
OUTPUT_ASIG_TXT = root_path("hectarea_asignacion.txt")
AREA_HA = 1.0  # una hectárea
MODO = "bootstrap"  # "bootstrap" | "media"
RANDOM_SEED = None  # fija un entero para reproducibilidad


def cargar_densidades(csv_path: str):
    especies: List[str] = []
    conteos_por_poligono: List[np.ndarray] = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        filas = list(reader)

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

    densidades: List[np.ndarray] = []
    for row in filas:
        nombre = row[0]
        if "Área" in nombre:
            continue
        conteos = np.array([float(x) for x in row[1:]], dtype=float)
        dens = conteos / areas
        especies.append(nombre)
        conteos_por_poligono.append(conteos)
        densidades.append(dens)

    return especies, np.array(densidades, dtype=float)


def calcular_conteos_1ha(densidades: np.ndarray, modo: str = MODO, rng: np.random.Generator | None = None) -> np.ndarray:
    """Retorna enteros de plantas por especie para 1 Ha.
    densidades: shape (n_tipos, n_poligonos)
    """
    n_tipos = densidades.shape[0]
    if rng is None:
        rng = np.random.default_rng()

    if modo == "media":
        dens = densidades.mean(axis=1)
    elif modo == "bootstrap":
        # elegir una densidad observada al azar para cada especie (con reemplazo)
        idx = rng.integers(low=0, high=densidades.shape[1], size=n_tipos)
        dens = densidades[np.arange(n_tipos), idx]
    else:
        raise ValueError("modo debe ser 'media' o 'bootstrap'")

    # conteos por 1 Ha
    counts_float = dens * AREA_HA
    # convertir a enteros: redondeo al entero más cercano
    counts = np.rint(counts_float).astype(int)
    # asegurar no negativos
    counts = np.maximum(counts, 0)
    return counts


def construir_asignacion_completa(counts: np.ndarray, n_nodos: int, rng: np.random.Generator | None = None) -> List[int]:
    """Devuelve una lista de largo n_nodos con índices de especie y -1 para vacíos."""
    if rng is None:
        rng = np.random.default_rng()
    asignados = []
    for i, c in enumerate(counts):
        asignados.extend([i] * int(c))
    total = int(sum(counts))
    vacios = max(0, n_nodos - total)
    asignados.extend([-1] * vacios)
    asignados = np.array(asignados, dtype=int)
    # barajar posiciones
    rng.shuffle(asignados)
    # si por alguna razón hay más que n_nodos (no debería), recortar
    if len(asignados) > n_nodos:
        asignados = asignados[:n_nodos]
    # si hay menos (raro), rellenar con vacíos
    if len(asignados) < n_nodos:
        faltan = n_nodos - len(asignados)
        asignados = np.concatenate([asignados, np.full(faltan, -1, dtype=int)])
        rng.shuffle(asignados)
    return asignados.tolist()


def imprimir_formato(counts: np.ndarray, especies: List[str]) -> str:
    lineas = []
    lineas.append("1")
    for i, c in enumerate(counts):
        lineas.append(f"{especies[i]}: {int(c)}")
    lineas.append("")
    lineas.append("indice del espacio: que planta está")
    return "\n".join(lineas)

def guardar_asignacion_txt(asignacion: List[int], especies: List[str], path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write("indice del espacio: que planta está\n")
        for idx, val in enumerate(asignacion):
            nombre = "vacío" if val == -1 else especies[val]
            f.write(f"{idx}: {nombre}\n")


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    # Cargar densidades y grafo para conocer n_nodos (espacios)
    especies, densidades = cargar_densidades(CSV_PATH)
    counts = calcular_conteos_1ha(densidades, modo=MODO, rng=rng)
    # Determinar número de espacios del terreno (usando grafo)
    with open(data_path("grafo_hexagonal.json"), "r") as fg:
        grafo = json.load(fg)
    n_nodos = len(grafo.get("nodes", []))
    asignacion = construir_asignacion_completa(counts, n_nodos=n_nodos, rng=rng)

    # Guardar JSON
    data = {
        "area_ha": AREA_HA,
        "modo": MODO,
        "n_nodos": n_nodos,
        "especies": especies,
        "conteos_por_especie": {especies[i]: int(counts[i]) for i in range(len(especies))},
        "total_plantas": int(sum(counts)),
        "asignacion_indices": asignacion,  # largo n_nodos; -1=vacío
        "asignacion_nombres": ["vacío" if a == -1 else especies[a] for a in asignacion],
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Guardar TXT con el formato solicitado (sin listar toda la asignación para no inflar el archivo)
    texto = imprimir_formato(counts, especies)
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(texto + "\n")
    # Guardar asignación completa con vacíos
    guardar_asignacion_txt(asignacion, especies, OUTPUT_ASIG_TXT)

    print(texto)
    print(f"\nTotal de plantas en 1 Ha: {int(sum(counts))}")
    print(f"Salidas: {OUTPUT_JSON}, {OUTPUT_TXT}, {OUTPUT_ASIG_TXT}")


if __name__ == "__main__":
    main()
