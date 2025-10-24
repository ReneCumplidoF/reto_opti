from __future__ import annotations

"""
Algoritmo Genético para optimizar la cantidad inicial de plantas (10 especies).

Chromosoma: vector entero de longitud 10 con conteos por especie.
Restricciones:
- Cada gen debe estar entre [floor(0.9*b), ceil(1.1*b)] donde b son los conteos base
- La suma total debe ser exactamente 658

Fitness: costo mínimo (competencia) obtenido al correr utils.hormigas.optimizar_aco
		 utilizando el cromosoma como override de plantas_totales.

Parada: tiempo máximo (por defecto 45 minutos). Se termina la generación en curso,
		se guardan archivos con la mejor solución y se imprime el resultado.

Archivos de salida:
- output/mejor_cromosoma_genetico.npy: vector (10,) con el mejor cromosoma
- output/historial_costos_ga.npy: mejor fitness por generación (menor es mejor)
- output/mejor_asignacion_hormigas.npy y output/historial_costos_aco.npy
  (generados por ACO para el mejor cromosoma)
"""

from dataclasses import dataclass
import time
import math
import json
import sys
from pathlib import Path
from typing import Callable, List, Tuple, Dict

import numpy as np

# Agregar el directorio raíz al path para imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.paths import output_path
from utils.hormigas import optimizar_aco


# ----- Configuración base y límites -----
BASE_COUNTS = np.array([42, 196, 42, 42, 49, 38, 73, 64, 86, 26], dtype=int)
SUMA_OBJETIVO = 658
LB = np.floor(0.9 * BASE_COUNTS).astype(int)
UB = np.ceil(1.1 * BASE_COUNTS).astype(int)


def reparar_vector_sum_bounds(vec: np.ndarray, suma_obj: int, lb: np.ndarray, ub: np.ndarray, rng: np.random.Generator) -> np.ndarray:
	"""Repara un vector entero para cumplir límites por gen y suma total exacta.

	Estrategia: recortar a [lb, ub] y ajustar el déficit/exceso distribuyendo
	aleatoriamente sin violar los límites.
	"""
	v = np.clip(vec.astype(int), lb, ub).copy()
	diff = int(suma_obj - v.sum())
	if diff == 0:
		return v

	idx = np.arange(v.size)
	# Ajuste incremental/decremental pequeño para evitar grandes saltos
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
		else:  # diff < 0
			for i in idx:
				if v[i] > lb[i]:
					v[i] -= 1
					diff += 1
					changed = True
					if diff == 0:
						break
		if not changed:
			# No podemos cambiar más sin violar límites: salir
			break
	return v


def init_poblacion(n: int, rng: np.random.Generator) -> np.ndarray:
	poblacion = []
	for _ in range(n):
		# muestreo uniforme por gen dentro de [lb, ub]
		vec = rng.integers(LB, UB + 1)
		vec = reparar_vector_sum_bounds(vec, SUMA_OBJETIVO, LB, UB, rng)
		poblacion.append(vec)
	return np.array(poblacion, dtype=int)


def torneo_indices(fitness: np.ndarray, k: int, rng: np.random.Generator) -> int:
	cand = rng.integers(0, fitness.size, size=k)
	# menor fitness es mejor
	best = cand[np.argmin(fitness[cand])]
	return int(best)


def crossover_un_punto(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
	if p1.size != p2.size:
		raise ValueError("Padres con longitud distinta")
	n = p1.size
	if n <= 1:
		return p1.copy(), p2.copy()
	punto = rng.integers(1, n)  # corte entre [1, n-1]
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


def evaluar_cromosoma(crom: np.ndarray, aco_params: Dict, cache: Dict[Tuple[int, ...], float], rng_seed: int | None = None) -> float:
	key = tuple(int(x) for x in crom)
	if key in cache:
		return cache[key]

	# Llamar ACO con override de plantas_totales
	res = optimizar_aco(
		plantas_totales=crom,
		guardar_resultados=False,
		verbose=False,
		**aco_params,
	)
	fit = float(res["mejor_costo"]) if np.isfinite(res["mejor_costo"]) else float("inf")
	cache[key] = fit
	return fit


@dataclass
class GAParams:
	pop_size: int = 40
	torneo_k: int = 3
	p_cruza: float = 0.9
	p_mut: float = 0.2
	elitismo: int = 2
	tiempo_max_min: float = 45.0
	# Parámetros ACO por evaluación (ajustables)
	aco_hormigas: int = 25
	aco_iter: int = 25
	aco_alfa: float = 1.0
	aco_beta: float = 2.0
	aco_rho: float = 0.1
	aco_Q: float = 1.0
	aco_elitismo: int = 5
	aco_max_intentos_busqueda: int = 200
	seed: int | None = None


def ejecutar_ga(params: GAParams) -> Dict:
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

	t0 = time.time()
	generacion = 0
	mejor_fit = float("inf")
	mejor_crom = None

	# Ejecutar hasta tiempo máximo, cerrando generación en curso
	while True:
		# Evaluar población
		fitness = np.array([evaluar_cromosoma(ind, aco_params, eval_cache) for ind in poblacion], dtype=float)

		# Actualizar mejor global
		idx_best = int(np.argmin(fitness))
		if fitness[idx_best] < mejor_fit:
			mejor_fit = float(fitness[idx_best])
			mejor_crom = poblacion[idx_best].copy()

		historial_mejor.append(mejor_fit)
		generacion += 1

		# ¿Se excedió el tiempo? si sí, terminar tras cerrar esta generación
		if (time.time() - t0) >= params.tiempo_max_min * 60.0:
			break

		# Elitismo
		orden = np.argsort(fitness)
		nuevos = [poblacion[i].copy() for i in orden[: params.elitismo]]

		# Reproducción
		while len(nuevos) < params.pop_size:
			i1 = torneo_indices(fitness, params.torneo_k, rng)
			i2 = torneo_indices(fitness, params.torneo_k, rng)
			p1 = poblacion[i1]
			p2 = poblacion[i2]

			if rng.random() < params.p_cruza:
				h1, h2 = crossover_un_punto(p1, p2, rng)
			else:
				h1, h2 = p1.copy(), p2.copy()

			# Mutación y reparación
			h1 = mutacion_escalar(h1, params.p_mut, rng)
			h1 = reparar_vector_sum_bounds(h1, SUMA_OBJETIVO, LB, UB, rng)

			if len(nuevos) + 1 < params.pop_size:
				h2 = mutacion_escalar(h2, params.p_mut, rng)
				h2 = reparar_vector_sum_bounds(h2, SUMA_OBJETIVO, LB, UB, rng)
				nuevos.extend([h1, h2])
			else:
				nuevos.append(h1)

		poblacion = np.array(nuevos, dtype=int)

	# Re-evaluar el mejor cromosoma ejecutando ACO con guardado de resultados
	assert mejor_crom is not None
	resultado_aco = optimizar_aco(
		plantas_totales=mejor_crom,
		guardar_resultados=True,
		verbose=True,
		n_hormigas=params.aco_hormigas,
		n_iter=params.aco_iter,
		alfa=params.aco_alfa,
		beta=params.aco_beta,
		rho=params.aco_rho,
		Q=params.aco_Q,
		elitismo=params.aco_elitismo,
		max_intentos_busqueda=params.aco_max_intentos_busqueda,
	)

	# Guardar artefactos del GA
	np.save(output_path("mejor_cromosoma_genetico.npy"), mejor_crom)
	np.save(output_path("historial_costos_ga.npy"), np.array(historial_mejor))

	summary = {
		"mejor_costo": float(mejor_fit),
		"mejor_cromosoma": mejor_crom.tolist(),
		"generaciones": generacion,
		"tiempo_seg": round(time.time() - t0, 2),
	}
	with open(output_path("resumen_ga.json"), "w", encoding="utf-8") as fh:
		json.dump(summary, fh, ensure_ascii=False, indent=2)

	# Mostrar resumen
	print("\n================ GA COMPLETADO ================")
	print(f"Tiempo total: {summary['tiempo_seg']} s | Generaciones: {generacion}")
	print(f"Mejor costo (ACO): {mejor_fit:.4f}")
	print(f"Mejor cromosoma (10): {mejor_crom.tolist()}")
	print(f"Guardado: \n- {output_path('mejor_cromosoma_genetico.npy')}\n- {output_path('historial_costos_ga.npy')}\n- {output_path('resumen_ga.json')}")

	return {
		"mejor_cromosoma": mejor_crom,
		"mejor_costo": mejor_fit,
		"historial": historial_mejor,
		"resultado_aco": resultado_aco,
	}


if __name__ == "__main__":
	# Ejecutar con parámetros por defecto (45 minutos). Ajusta con env o edita aquí.
	params = GAParams()
	ejecutar_ga(params)

