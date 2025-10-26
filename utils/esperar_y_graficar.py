"""
Espera a que termine la corrida ACO (detectando la creación del archivo resumen_aco_best.json)
y luego ejecuta el generador de gráficos utils/graficos_insights.py.

Uso:
  python -m utils.esperar_y_graficar
o
  python utils/esperar_y_graficar.py
"""
from __future__ import annotations

import time
from pathlib import Path
import sys

# Asegurar imports locales
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import output_path


def main(poll_seconds: int = 30, timeout_seconds: int | None = None) -> None:
    objetivo = output_path("resumen_aco_best.json")
    inicio = time.time()
    print(f"Watcher: esperando a {objetivo} ... (poll={poll_seconds}s)")
    while True:
        if objetivo.exists():
            # Dar un pequeño margen para que el escritor termine
            time.sleep(5)
            print("Archivo detectado. Generando gráficos...")
            break
        if timeout_seconds is not None and (time.time() - inicio) > timeout_seconds:
            print("Timeout alcanzado; no se generaron gráficos.")
            return
        time.sleep(poll_seconds)

    # Ejecutar generador de gráficos
    try:
        from utils.graficos_insights import main as gen_main
        gen_main()
    except Exception as e:
        print("Error generando gráficos:", e)


if __name__ == "__main__":
    main()
