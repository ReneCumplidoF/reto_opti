"""
Genera historial sintético de cromosomas basado en los datos actuales para visualización.
Usa el mejor cromosoma actual como punto final y simula una convergencia gradual desde el baseline.

Esto es útil para visualizar la corrida pasada. Las futuras corridas del GA generarán
el historial real automáticamente.
"""
from __future__ import annotations

from pathlib import Path
import sys
import json

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import output_path

BASELINE = np.array([42, 196, 42, 42, 49, 38, 73, 64, 86, 26])


def reconstruir_historial():
    """Reconstruye un historial sintético para visualización."""
    
    # Cargar datos reales
    p_resumen = output_path("resumen_ga.json")
    p_mejor = output_path("mejor_cromosoma_genetico.npy")
    
    if not p_resumen.exists() or not p_mejor.exists():
        print("❌ Faltan archivos. Ejecuta el GA primero.")
        return
    
    with open(p_resumen, 'r') as fh:
        resumen = json.load(fh)
    
    mejor = np.load(p_mejor)
    n_gens = resumen.get('generaciones', 59)
    
    print(f"Reconstruyendo historial para {n_gens} generaciones...")
    
    # Crear interpolación suave del baseline al mejor
    historial = np.zeros((n_gens, 10))
    
    for i in range(10):  # Por cada especie
        # Interpolación no lineal (más rápido al principio, luego estabiliza)
        t = np.linspace(0, 1, n_gens)
        # Función sigmoide suavizada para simular convergencia
        decay = 1 - np.exp(-3 * t)  # Converge rápido al principio
        historial[:, i] = BASELINE[i] + (mejor[i] - BASELINE[i]) * decay
    
    # Agregar pequeñas variaciones aleatorias para simular la búsqueda
    rng = np.random.RandomState(42)
    for gen in range(n_gens):
        factor_ruido = max(0, 1 - gen / n_gens) * 2  # Más ruido al principio
        ruido = rng.randn(10) * factor_ruido
        historial[gen] = np.round(historial[gen] + ruido).astype(int)
    
    # Asegurar suma = 658 en cada generación
    for gen in range(n_gens):
        suma = historial[gen].sum()
        if suma != 658:
            diff = 658 - suma
            # Distribuir diferencia en especies con más margen
            indices = rng.choice(10, abs(diff), replace=True)
            for idx in indices:
                historial[gen, idx] += np.sign(diff)
    
    # Última generación debe ser exactamente el mejor
    historial[-1] = mejor
    
    # Guardar
    np.save(output_path("historial_cromosomas.npy"), historial)
    print(f"✓ Historial reconstruido guardado en: {output_path('historial_cromosomas.npy')}")
    print(f"  Shape: {historial.shape}")
    print(f"  Primera gen: {historial[0].tolist()}")
    print(f"  Última gen:  {historial[-1].tolist()}")


if __name__ == "__main__":
    reconstruir_historial()
