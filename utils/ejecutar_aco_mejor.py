"""
Ejecuta ACO (100 hormigas × 100 iteraciones) usando el mejor cromosoma guardado por el GA.

Lee:  output/mejor_cromosoma_genetico.npy
Escribe: 
  - output/mejor_asignacion_hormigas.npy
  - output/historial_costos_aco.npy
  - output/resumen_aco_best.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path
import sys

import numpy as np

# Asegurar imports de paquete local
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import output_path
from utils.hormigas import optimizar_aco


def main():
    p_mejor = output_path("mejor_cromosoma_genetico.npy")
    if not (p_mejor.exists() and p_mejor.is_file()):
        raise FileNotFoundError(f"No se encontró {p_mejor}. Ejecuta el GA primero para generar el mejor cromosoma.")

    mejor = np.load(p_mejor)
    print("Mejor cromosoma cargado:", mejor.tolist(), "| suma=", int(mejor.sum()))

    t0 = time.time()
    print("\n▶ Ejecutando ACO con 100 hormigas × 100 iteraciones...")
    res = optimizar_aco(
        plantas_totales=mejor,
        n_hormigas=100,
        n_iter=100,
        guardar_resultados=True,
        verbose=True,
    )
    dt = round(time.time() - t0, 2)
    print("\n✔ ACO finalizado. Tiempo:", dt, "s | Mejor costo:", res.get("mejor_costo"))

    resumen = {
        "n_hormigas": 100,
        "n_iter": 100,
        "mejor_costo": float(res.get("mejor_costo", float("nan"))),
        "tiempo_seg": dt,
        "mejor_cromosoma": mejor.tolist(),
    }
    with open(output_path("resumen_aco_best.json"), "w", encoding="utf-8") as fh:
        json.dump(resumen, fh, ensure_ascii=False, indent=2)
    print("Resumen guardado en:", output_path("resumen_aco_best.json"))


if __name__ == "__main__":
    main()
