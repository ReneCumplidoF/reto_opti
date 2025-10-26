"""
Versión paralela del GA usando multiprocessing.
Evalúa múltiples cromosomas simultáneamente en diferentes cores.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
import json
import sys
from pathlib import Path
from typing import Tuple, Dict
from multiprocessing import Pool, cpu_count

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.paths import output_path
from utils.hormigas import optimizar_aco

# Configuración base
BASE_COUNTS = np.array([42, 196, 42, 42, 49, 38, 73, 64, 86, 26], dtype=int)
SUMA_OBJETIVO = 658
LB = np.floor(0.9 * BASE_COUNTS).astype(int)
UB = np.ceil(1.1 * BASE_COUNTS).astype(int)


def reparar_vector_sum_bounds(vec: np.ndarray, suma_obj: int, lb: np.ndarray, ub: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    v = np.clip(vec.astype(int), lb, ub).copy()
    diff = int(suma_obj - v.sum())
    if diff == 0:
        return v

    idx = np.arange(v.size)
    while diff != 0:
        rng.shuffle(idx)
        changed = False
        if diff > 0:
            for i in idx:
                if v[i] < ub[i]:
                    v[i] += 1
                    diff -= 1
                    changed = True
                    if diff == 0:
                        break
        else:
            for i in idx:
                if v[i] > lb[i]:
                    v[i] -= 1
                    diff += 1
                    changed = True
                    if diff == 0:
                        break
        if not changed:
            break
    return v


def init_poblacion(n: int, rng: np.random.Generator) -> np.ndarray:
    poblacion = []
    for _ in range(n):
        vec = rng.integers(LB, UB + 1)
        vec = reparar_vector_sum_bounds(vec, SUMA_OBJETIVO, LB, UB, rng)
        poblacion.append(vec)
    return np.array(poblacion, dtype=int)


def torneo_indices(fitness: np.ndarray, k: int, rng: np.random.Generator) -> int:
    cand = rng.integers(0, fitness.size, size=k)
    best = cand[np.argmin(fitness[cand])]
    return int(best)


def crossover_un_punto(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    n = p1.size
    if n <= 1:
        return p1.copy(), p2.copy()
    punto = rng.integers(1, n)
    h1 = np.concatenate([p1[:punto], p2[punto:]])
    h2 = np.concatenate([p2[:punto], p1[punto:]])
    return h1, h2


def mutacion_escalar(v: np.ndarray, prob: float, rng: np.random.Generator, paso_max: int = 3) -> np.ndarray:
    u = v.copy()
    for i in range(u.size):
        if rng.random() < prob:
            delta = int(rng.integers(-paso_max, paso_max + 1))
            if delta == 0:
                delta = 1 if rng.random() < 0.5 else -1
            u[i] = u[i] + delta
    return u


# Función de evaluación para paralelizar
def evaluar_cromosoma_worker(args):
    """Worker function para evaluación paralela."""
    crom, aco_params = args
    res = optimizar_aco(
        plantas_totales=crom,
        guardar_resultados=False,
        verbose=False,
        **aco_params,
    )
    fit = float(res["mejor_costo"]) if np.isfinite(res["mejor_costo"]) else float("inf")
    return tuple(int(x) for x in crom), fit


@dataclass
class GAParamsParalelo:
    pop_size: int = 40
    torneo_k: int = 3
    p_cruza: float = 0.9
    p_mut: float = 0.2
    elitismo: int = 2
    tiempo_max_min: float = 45.0
    n_workers: int = None  # None = usar todos los cores disponibles
    early_stopping_gens: int = 4  # Parar si no mejora en N generaciones (0 = desactivado)
    # Parámetros ACO
    aco_hormigas: int = 15
    aco_iter: int = 15
    aco_alfa: float = 1.0
    aco_beta: float = 2.0
    aco_rho: float = 0.1
    aco_Q: float = 1.0
    aco_elitismo: int = 5
    aco_max_intentos_busqueda: int = 200
    seed: int | None = None


def ejecutar_ga_paralelo(params: GAParamsParalelo) -> Dict:
    rng = np.random.default_rng(params.seed)
    poblacion = init_poblacion(params.pop_size, rng)
    historial_mejor = []
    eval_cache: Dict[Tuple[int, ...], float] = {}

    aco_params = dict(
        n_hormigas=params.aco_hormigas,
        n_iter=params.aco_iter,
        alfa=params.aco_alfa,
        beta=params.aco_beta,
        rho=params.aco_rho,
        Q=params.aco_Q,
        elitismo=params.aco_elitismo,
        max_intentos_busqueda=params.aco_max_intentos_busqueda,
    )

    # Determinar número de workers
    n_workers = params.n_workers if params.n_workers else cpu_count()
    print(f"🚀 Usando {n_workers} workers en paralelo (CPU cores disponibles: {cpu_count()})")

    t0 = time.time()
    generacion = 0
    mejor_fit = float("inf")
    mejor_crom = None
    generaciones_sin_mejora = 0  # Contador para early stopping
    interrumpido = False  # Ctrl+C
    historial_cromosomas = []  # Guardar mejor cromosoma por generación

    with Pool(processes=n_workers) as pool:
        try:
            while True:
                # Separar individuos ya evaluados de los nuevos
                nuevos_a_evaluar = []
                indices_nuevos = []
                fitness_parcial = np.full(len(poblacion), np.inf)

                for idx, ind in enumerate(poblacion):
                    key = tuple(int(x) for x in ind)
                    if key in eval_cache:
                        fitness_parcial[idx] = eval_cache[key]
                    else:
                        nuevos_a_evaluar.append((ind, aco_params))
                        indices_nuevos.append(idx)

                # Evaluar nuevos en paralelo
                if nuevos_a_evaluar:
                    resultados = pool.map(evaluar_cromosoma_worker, nuevos_a_evaluar)
                    for idx_relativo, (key, fit) in enumerate(resultados):
                        idx_absoluto = indices_nuevos[idx_relativo]
                        eval_cache[key] = fit
                        fitness_parcial[idx_absoluto] = fit

                fitness = fitness_parcial

                # Actualizar mejor global
                idx_best = int(np.argmin(fitness))
                if fitness[idx_best] < mejor_fit:
                    mejor_fit = float(fitness[idx_best])
                    mejor_crom = poblacion[idx_best].copy()
                    generaciones_sin_mejora = 0  # Reiniciar contador
                else:
                    generaciones_sin_mejora += 1

                historial_mejor.append(mejor_fit)
                historial_cromosomas.append(mejor_crom.copy())  # Guardar cromosoma
                generacion += 1

                print(f"Gen {generacion:3d} | Mejor: {mejor_fit:.4f} | Evaluaciones nuevas: {len(nuevos_a_evaluar):2d} | Sin mejora: {generaciones_sin_mejora} | Tiempo: {time.time()-t0:.1f}s")

                # Verificar señal por archivo (parada suave al terminar la generación)
                try:
                    stop_file = output_path("GA_STOP")
                    if stop_file.exists():
                        print(f"\n🛑 Señal de parada detectada: {stop_file}. Finalizando tras esta generación...")
                        break
                except Exception:
                    # No bloquear por errores de FS
                    pass

                # Verificar early stopping
                if params.early_stopping_gens > 0 and generaciones_sin_mejora >= params.early_stopping_gens:
                    print(f"\n⏸️  Early stopping: No hubo mejora en {params.early_stopping_gens} generaciones consecutivas")
                    break

                # Verificar tiempo
                if (time.time() - t0) >= params.tiempo_max_min * 60.0:
                    print(f"\n⏰ Tiempo límite alcanzado: {params.tiempo_max_min} minutos")
                    break

                # Generar nueva población
                orden = np.argsort(fitness)
                nuevos = [poblacion[i].copy() for i in orden[: params.elitismo]]

                while len(nuevos) < params.pop_size:
                    i1 = torneo_indices(fitness, params.torneo_k, rng)
                    i2 = torneo_indices(fitness, params.torneo_k, rng)
                    p1 = poblacion[i1]
                    p2 = poblacion[i2]

                    if rng.random() < params.p_cruza:
                        h1, h2 = crossover_un_punto(p1, p2, rng)
                    else:
                        h1, h2 = p1.copy(), p2.copy()

                    h1 = mutacion_escalar(h1, params.p_mut, rng)
                    h1 = reparar_vector_sum_bounds(h1, SUMA_OBJETIVO, LB, UB, rng)

                    if len(nuevos) + 1 < params.pop_size:
                        h2 = mutacion_escalar(h2, params.p_mut, rng)
                        h2 = reparar_vector_sum_bounds(h2, SUMA_OBJETIVO, LB, UB, rng)
                        nuevos.extend([h1, h2])
                    else:
                        nuevos.append(h1)

                poblacion = np.array(nuevos, dtype=int)

        except KeyboardInterrupt:
            interrumpido = True
            print("\n🛑 Interrupción (Ctrl+C) recibida. Guardando el mejor resultado hasta ahora...")

    # Re-evaluar mejor cromosoma con guardado
    print("\n🔄 Re-evaluando mejor cromosoma con guardado completo...")
    assert mejor_crom is not None
    resultado_aco = optimizar_aco(
        plantas_totales=mejor_crom,
        guardar_resultados=True,
        verbose=True,
        **aco_params,
    )

    # Guardar resultados
    np.save(output_path("mejor_cromosoma_genetico.npy"), mejor_crom)
    np.save(output_path("historial_costos_ga.npy"), np.array(historial_mejor))
    np.save(output_path("historial_cromosomas.npy"), np.array(historial_cromosomas))

    summary = {
        "mejor_costo": float(mejor_fit),
        "mejor_cromosoma": mejor_crom.tolist(),
        "generaciones": generacion,
        "tiempo_seg": round(time.time() - t0, 2),
        "n_workers": n_workers,
        "early_stopped": generaciones_sin_mejora >= params.early_stopping_gens if params.early_stopping_gens > 0 else False,
        "generaciones_sin_mejora_final": generaciones_sin_mejora,
        "interrumpido": interrumpido,
    }
    with open(output_path("resumen_ga.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print("\n" + "="*60)
    print("================ GA PARALELO COMPLETADO ================")
    print("="*60)
    print(f"Tiempo total: {summary['tiempo_seg']} s | Generaciones: {generacion}")
    print(f"Workers usados: {n_workers} cores")
    if summary['early_stopped']:
        print(f"⏸️  Detenido por convergencia (sin mejora en {params.early_stopping_gens} gens)")
    print(f"Mejor costo (ACO): {mejor_fit:.4f}")
    print(f"Mejor cromosoma: {mejor_crom.tolist()}")
    print(f"\nArchivos guardados en output/:")
    print(f"  - mejor_cromosoma_genetico.npy")
    print(f"  - historial_costos_ga.npy")
    print(f"  - historial_cromosomas.npy")
    print(f"  - resumen_ga.json")
    print("="*60)

    return {
        "mejor_cromosoma": mejor_crom,
        "mejor_costo": mejor_fit,
        "historial": historial_mejor,
        "generaciones": generacion,
        "tiempo_seg": round(time.time() - t0, 2),
        "resultado_aco": resultado_aco,
    }


if __name__ == "__main__":
    # Configuración optimizada para CPU multicore
    params = GAParamsParalelo(
        pop_size=40,
        tiempo_max_min=45.0,
        n_workers=None,  # Usar todos los cores
        early_stopping_gens=4,  # Parar si no mejora en 4 generaciones
        aco_hormigas=15,  # Reducido para más generaciones
        aco_iter=15,
        seed=42
    )
    
    print(f"🎯 Iniciando GA Paralelo")
    print(f"📊 Configuración:")
    print(f"   - Población: {params.pop_size}")
    print(f"   - Tiempo límite: {params.tiempo_max_min} min")
    print(f"   - Early stopping: {params.early_stopping_gens} gens sin mejora")
    print(f"   - ACO: {params.aco_hormigas}×{params.aco_iter}")
    print(f"   - Cores disponibles: {cpu_count()}")
    print()
    
    ejecutar_ga_paralelo(params)
