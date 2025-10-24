"""
Algoritmo de Optimización por Colonia de Hormigas (ACO)
para asignación óptima de plantas minimizando competencia.

Uso:
    from utils.hormigas import optimizar_aco
    
    resultado = optimizar_aco(
        n_hormigas=40,
        n_iter=100,
        alfa=1.0,
        beta=2.0,
        rho=0.1,
        Q=1.0,
        elitismo=5,
        max_intentos_busqueda=200,
        verbose=True
    )
    
    mejor_asignacion = resultado['mejor_asignacion']
    mejor_costo = resultado['mejor_costo']
    historial = resultado['historial_costos']
"""

import numpy as np
import json
import csv
from collections import defaultdict
from pathlib import Path
import sys
from typing import Dict, Tuple, Optional, List

# Agregar el directorio raíz al path para imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import data_path, output_path


def cargar_totales_desde_csv(csv_path: str) -> np.ndarray:
    """Carga totales de plantas por especie desde CSV."""
    totales = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            nombre = row[0]
            if "Área" in nombre:
                continue  # ignorar fila de áreas
            # tomar columnas P1..Pn como enteros
            valores = [int(float(x)) for x in row[1:] if x.strip() != ""]
            totales.append(sum(valores))
    return np.array(totales, dtype=float)


def ajustar_restantes(restantes: np.ndarray, n_obj: int, pesos: np.ndarray) -> np.ndarray:
    """Ajusta conteos restantes para que sumen exactamente n_obj respetando proporciones."""
    r = restantes.astype(int).copy()
    r[r < 0] = 0
    total = int(r.sum())
    if total == n_obj:
        return r
    if n_obj <= 0:
        return np.zeros_like(r, dtype=int)
    if total < n_obj:
        deficit = n_obj - total
        # distribuir segun pesos (proporciones)
        orden = np.argsort(-pesos)
        idx = 0
        while deficit > 0:
            r[orden[idx % len(orden)]] += 1
            idx += 1
            deficit -= 1
    else:
        # total > n_obj -> recortar donde menos peso
        exceso = total - n_obj
        orden = np.argsort(pesos)  # de menor a mayor
        idx = 0
        while exceso > 0:
            j = orden[idx % len(orden)]
            if r[j] > 0:
                r[j] -= 1
                exceso -= 1
            idx += 1
    return r


def calcular_heuristica(nodo: int, planta: int, asignacion_parcial: np.ndarray, 
                       vecinos: Dict[int, List[int]], competencia: np.ndarray) -> float:
    """Calcula la heurística para asignar un tipo a un nodo.

    La heurística considera la competencia con vecinos ya asignados.
    Como queremos MINIMIZAR competencia, retornamos 1/(1+competencia)
    para que baja competencia = alta heurística.
    """
    suma_competencia = 0.0
    vecinos_asignados = 0

    for vecino in vecinos[nodo]:
        # Si el vecino ya tiene tipo asignado
        if asignacion_parcial[vecino].sum() > 0:  # vecino ya asignado
            tipo_vecino = np.argmax(asignacion_parcial[vecino])
            # Sumar la competencia (valores altos son peores)
            suma_competencia += competencia[planta, tipo_vecino]
            vecinos_asignados += 1

    # Retornar 1/(1+competencia) para que baja competencia = alta heurística
    return 1.0 / (1.0 + suma_competencia)


def costo(asignacion: np.ndarray, edges: List[List[int]], competencia: np.ndarray) -> float:
    """Calcula el costo total (competencia) de una asignación."""
    costo_total = 0.0
    for i, j in edges:
        tipo_i = np.argmax(asignacion[i])
        tipo_j = np.argmax(asignacion[j])
        # Queremos MINIMIZAR competencia
        costo_total += competencia[tipo_i, tipo_j]
    return costo_total


def busqueda_local_vecindad(
    asignacion: np.ndarray,
    edges: List[List[int]],
    vecinos: Dict[int, List[int]],
    competencia: np.ndarray,
    fijos_mask: np.ndarray,
    indices_libres: np.ndarray,
    n_nodos: int,
    n_tipos: int,
    max_intentos: int = 200,
) -> Tuple[np.ndarray, float]:
    """Búsqueda local optimizada: solo intercambia nodos libres (no fijos)."""
    mejor_asignacion = asignacion.copy()
    mejor_costo = costo(mejor_asignacion, edges, competencia)

    # Convertir asignación a vector de tipos para acceso rápido
    tipos = np.argmax(mejor_asignacion, axis=1)

    intentos = 0
    mejoras_consecutivas = 0

    if indices_libres.size == 0:
        return mejor_asignacion, mejor_costo

    while intentos < max_intentos:
        intentos += 1

        # Seleccionar un nodo libre aleatorio
        nodo_i = np.random.choice(indices_libres)
        tipo_i = tipos[nodo_i]

        # Buscar un candidato entre vecinos libres o nodos libres aleatorios
        vecinos_libres = [v for v in vecinos[nodo_i] if not fijos_mask[v]]
        if len(vecinos_libres) > 0 and np.random.random() < 0.7:
            # 70% del tiempo: buscar entre vecinos directos libres
            nodo_j = np.random.choice(vecinos_libres)
        else:
            # 30% del tiempo: buscar en un nodo libre aleatorio
            nodo_j = np.random.choice(indices_libres)

        tipo_j = tipos[nodo_j]

        # Solo intercambiar si los tipos son diferentes
        if tipo_i == tipo_j:
            continue

        # Calcular el cambio en costo sin recalcular todo
        # (solo afecta a los vecinos de i y j)
        delta_costo = 0.0

        # Impacto de cambiar nodo_i de tipo_i a tipo_j
        for vecino in vecinos[nodo_i]:
            tipo_vecino = tipos[vecino]
            delta_costo += competencia[tipo_j, tipo_vecino]  # nueva competencia
            delta_costo -= competencia[tipo_i, tipo_vecino]  # quitar vieja

        # Impacto de cambiar nodo_j de tipo_j a tipo_i
        for vecino in vecinos[nodo_j]:
            if vecino == nodo_i:  # Ya contado arriba
                continue
            tipo_vecino = tipos[vecino]
            delta_costo += competencia[tipo_i, tipo_vecino]
            delta_costo -= competencia[tipo_j, tipo_vecino]

        # Si mejora (reduce competencia), aceptar el cambio
        if delta_costo < 0:  # Mejora (minimizamos competencia)
            tipos[nodo_i] = tipo_j
            tipos[nodo_j] = tipo_i
            mejor_costo += delta_costo
            mejoras_consecutivas += 1

            # Si encontramos muchas mejoras, continuar buscando
            if mejoras_consecutivas >= 10:
                intentos = max(0, intentos - 5)  # "resetear" un poco el contador
                mejoras_consecutivas = 0

    # Reconstruir matriz de asignación
    mejor_asignacion_final = np.zeros((n_nodos, n_tipos), dtype=int)
    for nodo in range(n_nodos):
        mejor_asignacion_final[nodo, tipos[nodo]] = 1

    return mejor_asignacion_final, mejor_costo


def optimizar_aco(
    n_hormigas: int = 40,
    n_iter: int = 100,
    alfa: float = 1.0,
    beta: float = 2.0,
    rho: float = 0.1,
    Q: float = 1.0,
    elitismo: int = 5,
    tau_min: float = 0.01,
    tau_max: float = 10.0,
    max_intentos_busqueda: int = 200,
    grafo_path: Optional[str] = None,
    competencia_path: Optional[str] = None,
    csv_path: Optional[str] = None,
    hectarea_path: Optional[str] = None,
    plantas_totales: Optional[List[int] | np.ndarray] = None,
    guardar_resultados: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Ejecuta el algoritmo ACO para optimizar la asignación de plantas minimizando competencia.
    
    Parámetros:
    -----------
    n_hormigas : int, default=40
        Número de hormigas (soluciones) por iteración
    n_iter : int, default=100
        Número de iteraciones del algoritmo
    alfa : float, default=1.0
        Importancia de la feromona (α)
    beta : float, default=2.0
        Importancia de la heurística (β)
    rho : float, default=0.1
        Tasa de evaporación de feromonas (ρ)
    Q : float, default=1.0
        Constante para depositar feromona
    elitismo : int, default=5
        Número de mejores soluciones que depositan feromona adicional
    tau_min : float, default=0.01
        Feromona mínima
    tau_max : float, default=10.0
        Feromona máxima
    max_intentos_busqueda : int, default=200
        Intentos máximos en búsqueda local
    grafo_path : str, optional
        Ruta al archivo JSON del grafo (default: datos/grafo_hexagonal.json)
    competencia_path : str, optional
        Ruta a la matriz de competencia (default: datos/matriz_competencia.npy)
    csv_path : str, optional
        Ruta al CSV con info de especies (default: datos/info_actual.csv)
    hectarea_path : str, optional
        Ruta a la asignación inicial (default: datos/hectarea.json)
    guardar_resultados : bool, default=True
        Si True, guarda resultados en output/
    verbose : bool, default=True
        Si True, muestra progreso
    
    Retorna:
    --------
    dict con:
        - 'mejor_asignacion': matriz (n_nodos × n_tipos) con la mejor asignación
        - 'mejor_costo': float, competencia mínima encontrada
        - 'historial_costos': lista con mejor costo por iteración
        - 'plantas_existentes_por_tipo': conteos por tipo
        - 'n_nodos': número de nodos
        - 'n_tipos': número de tipos de plantas
    """
    # Usar paths por defecto si no se especifican
    if grafo_path is None:
        grafo_path = data_path("grafo_hexagonal.json")
    if competencia_path is None:
        competencia_path = data_path("matriz_competencia.npy")
    if csv_path is None:
        csv_path = data_path("info_actual.csv")
    if hectarea_path is None:
        hectarea_path = data_path("hectarea.json")
    
    # Cargar grafo y matriz de competencia
    competencia = np.load(competencia_path)
    
    with open(grafo_path, "r") as f:
        grafo = json.load(f)
    
    edges = grafo["edges"]
    
    # Construir lista de adyacencia para acceso rápido
    vecinos = defaultdict(list)
    for i, j in edges:
        vecinos[i].append(j)
        vecinos[j].append(i)
    
    n_nodos = len(grafo["nodes"])
    
    # Cargar totales y escalar a la cantidad de nodos del grafo
    if plantas_totales is not None:
        plantas_existentes_por_tipo = np.array(plantas_totales, dtype=int)
        if plantas_existentes_por_tipo.ndim != 1:
            raise ValueError("plantas_totales debe ser un vector 1D de longitud n_tipos")
        if plantas_existentes_por_tipo.sum() != n_nodos:
            raise ValueError(
                f"La suma de plantas_totales debe ser exactamente {n_nodos}, recibió {plantas_existentes_por_tipo.sum()}"
            )
        n_tipos = int(plantas_existentes_por_tipo.size)
        # proporciones coherentes con override
        proporciones = plantas_existentes_por_tipo / max(1, plantas_existentes_por_tipo.sum())
    else:
        _totales = cargar_totales_desde_csv(csv_path)
        if _totales.sum() <= 0:
            raise ValueError("Los totales de plantas suman 0; verifique el archivo CSV.")

        proporciones = _totales / _totales.sum()
        objetivo_reales = proporciones * n_nodos
        plantas_existentes_por_tipo = np.floor(objetivo_reales).astype(int)
        resto = n_nodos - plantas_existentes_por_tipo.sum()

        if resto > 0:
            fracciones = objetivo_reales - np.floor(objetivo_reales)
            idx_orden = np.argsort(-fracciones)
            plantas_existentes_por_tipo[idx_orden[:resto]] += 1
        elif resto < 0:
            idx_orden = np.argsort(objetivo_reales - np.floor(objetivo_reales))
            plantas_existentes_por_tipo[idx_orden[:abs(resto)]] -= 1

        n_tipos = len(plantas_existentes_por_tipo)
    
    # Cargar asignación inicial y construir máscara de fijos
    asignacion_inicial_indices = None
    try:
        with open(hectarea_path, "r", encoding="utf-8") as fh:
            data_hect = json.load(fh)
            asignacion_inicial_indices = data_hect.get("asignacion_indices") or data_hect.get("asignacion")
            if asignacion_inicial_indices is not None:
                if len(asignacion_inicial_indices) != n_nodos:
                    if verbose:
                        print(f"Advertencia: asignación inicial no coincide con n_nodos={n_nodos}. Se ignorará.")
                    asignacion_inicial_indices = None
    except FileNotFoundError:
        if verbose:
            print("No se encontró asignación inicial. Todos los nodos serán libres.")
    
    fijos_mask = np.zeros(n_nodos, dtype=bool)
    conteo_inicial_por_tipo = np.zeros(n_tipos, dtype=int)
    asignacion_inicial_matriz = np.zeros((n_nodos, n_tipos), dtype=int)
    
    if asignacion_inicial_indices is not None:
        asig_arr = np.array(asignacion_inicial_indices, dtype=int)
        for nodo, tipo in enumerate(asig_arr):
            if tipo >= 0:
                fijos_mask[nodo] = True
                asignacion_inicial_matriz[nodo, tipo] = 1
                conteo_inicial_por_tipo[tipo] += 1
    
    indices_libres = np.where(~fijos_mask)[0]
    n_libres = indices_libres.size
    
    plantas_restantes_objetivo = plantas_existentes_por_tipo - conteo_inicial_por_tipo
    plantas_restantes_objetivo = ajustar_restantes(plantas_restantes_objetivo, n_libres, proporciones)
    
    # Inicializar matriz de feromonas
    feromonas = np.ones((n_nodos, n_tipos)) * 0.1
    
    # Variables para el algoritmo principal
    mejor_asignacion = None
    mejor_costo = float('inf')
    historial_costos = []
    
    if verbose:
        print("="*60)
        print("INICIANDO OPTIMIZACIÓN ACO")
        print("="*60)
        print(f"Parámetros: {n_hormigas} hormigas, {n_iter} iteraciones")
        print(f"α={alfa}, β={beta}, ρ={rho}, Q={Q}, elitismo={elitismo}")
        print(f"τ_min={tau_min}, τ_max={tau_max}")
        print(f"\nOBJETIVO: MINIMIZAR competencia total entre vecinos")
        print(f"Nodos totales: {n_nodos} | Tipos de plantas: {n_tipos}")
        print(f"Nodos fijos: {int(fijos_mask.sum())} | Nodos libres: {n_libres}")
        print(f"Conteo objetivo por tipo: {plantas_existentes_por_tipo.tolist()}")
        print(f"Conteo restante a asignar: {plantas_restantes_objetivo.tolist()}")
        print("="*60 + "\n")
    
    # Bucle principal ACO
    for iteracion in range(n_iter):
        soluciones = []
        costos_iter = []

    # Fase de construcción: cada hormiga construye una solución
        for h in range(n_hormigas):
            # Partir de la preasignación fija
            asignacion = asignacion_inicial_matriz.copy()
            plantas_restantes = plantas_restantes_objetivo.copy()

            # Orden aleatorio de visita solo de nodos libres
            orden_nodos = np.random.permutation(indices_libres)

            for i in orden_nodos:
                # Calcular heurística dinámica para cada tipo disponible
                heuristicas_tipos = np.zeros(n_tipos)
                for tipo in range(n_tipos):
                    if plantas_restantes[tipo] > 0:
                        heuristicas_tipos[tipo] = calcular_heuristica(i, tipo, asignacion, vecinos, competencia)

                # Probabilidad proporcional a feromona^alfa * heurística^beta
                probs = (feromonas[i] ** alfa) * (heuristicas_tipos ** beta)

                # Solo tipos con plantas disponibles
                probs = probs * (plantas_restantes > 0)

                if probs.sum() == 0:
                    # si no hay disponibilidad, elegir un tipo al azar con mayor heurística
                    tipo = np.argmax(heuristicas_tipos)
                else:
                    probs = probs / probs.sum()
                    tipo = np.random.choice(n_tipos, p=probs)

                asignacion[i, tipo] = 1
                plantas_restantes[tipo] -= 1

            # Búsqueda local para mejorar la solución (solo en nodos libres)
            if n_libres > 0:
                asignacion_mejorada, costo_mejorado = busqueda_local_vecindad(
                    asignacion,
                    edges,
                    vecinos,
                    competencia,
                    fijos_mask,
                    indices_libres,
                    n_nodos,
                    n_tipos,
                    max_intentos=max_intentos_busqueda,
                )
            else:
                asignacion_mejorada, costo_mejorado = asignacion, costo(asignacion, edges, competencia)

            soluciones.append(asignacion_mejorada)
            costos_iter.append(costo_mejorado)

    # Actualizar mejor solución global
        min_costo_iter = min(costos_iter) if len(costos_iter) > 0 else float('inf')
        if min_costo_iter < mejor_costo:
            mejor_costo = min_costo_iter
            idx_mejor = costos_iter.index(min_costo_iter)
            mejor_asignacion = soluciones[idx_mejor].copy()

        historial_costos.append(mejor_costo if np.isfinite(mejor_costo) else 0.0)

        # Evaporación de feromonas
        feromonas = (1 - rho) * feromonas

        # Depositar feromona (solo las mejores hormigas)
        indices_ordenados = np.argsort(costos_iter)

        for rank in range(min(elitismo, len(soluciones))):
            idx = indices_ordenados[rank]
            asignacion = soluciones[idx]
            costo_sol = costos_iter[idx]

            # Cantidad de feromona a depositar (inversamente proporcional al costo)
            delta_tau = Q / (1.0 + abs(costo_sol))

            # Depositar feromona en las asignaciones de esta solución
            for nodo in range(n_nodos):
                # No es necesario excluir fijos, pero podríamos hacerlo si se desea
                tipo = np.argmax(asignacion[nodo])
                feromonas[nodo, tipo] += delta_tau * (elitismo - rank)

        # Depositar feromona extra de la mejor solución global (elitismo global)
        if mejor_asignacion is not None and np.isfinite(mejor_costo):
            delta_tau_elite = Q / (1.0 + abs(mejor_costo))
            for nodo in range(n_nodos):
                tipo = np.argmax(mejor_asignacion[nodo])
                feromonas[nodo, tipo] += delta_tau_elite * elitismo * 2

        # Limitar feromonas entre tau_min y tau_max
        feromonas = np.clip(feromonas, tau_min, tau_max)

        # Mostrar progreso cada 10 iteraciones
        if verbose and (iteracion + 1) % 10 == 0:
            prom = np.mean(costos_iter) if len(costos_iter) > 0 else float('nan')
            print(f"Iteración {iteracion+1}/{n_iter} | Mejor: {mejor_costo:.4f} | Promedio: {prom:.4f}")
    
    if verbose:
        print("\n" + "="*60)
        print("OPTIMIZACIÓN COMPLETADA")
        print("="*60)
        print(f"Competencia total mínima encontrada: {mejor_costo:.4f}")
    
    # Guardar resultados si se solicita
    if guardar_resultados:
        np.save(output_path("mejor_asignacion_hormigas.npy"), mejor_asignacion)
        np.save(output_path("historial_costos_aco.npy"), np.array(historial_costos))
        if verbose:
            print(f"\n✓ Mejor asignación → {output_path('mejor_asignacion_hormigas.npy')}")
            print(f"✓ Historial costos → {output_path('historial_costos_aco.npy')}")
    
    # Retornar resultados
    return {
        'mejor_asignacion': mejor_asignacion,
        'mejor_costo': mejor_costo,
        'historial_costos': historial_costos,
        'plantas_existentes_por_tipo': plantas_existentes_por_tipo,
        'n_nodos': n_nodos,
        'n_tipos': n_tipos,
        'fijos_mask': fijos_mask,
        'indices_libres': indices_libres
    }


# Ejemplo de uso si se ejecuta directamente
if __name__ == "__main__":
    resultado = optimizar_aco(
        n_hormigas=40,
        n_iter=100,
        alfa=1.0,
        beta=2.0,
        rho=0.1,
        Q=1.0,
        elitismo=5,
        verbose=True
    )
    
    print("\n" + "="*60)
    print("RESUMEN DE RESULTADOS")
    print("="*60)
    print(f"Competencia mínima: {resultado['mejor_costo']:.4f}")
    print(f"Número de nodos: {resultado['n_nodos']}")
    print(f"Número de tipos: {resultado['n_tipos']}")
    print(f"Plantas por tipo: {resultado['plantas_existentes_por_tipo'].tolist()}")

