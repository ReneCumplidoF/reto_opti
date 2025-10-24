import numpy as np
import json
from collections import defaultdict

# Cargar grafo y matriz de plantas
grafo_path = "grafo_hexagonal.json"
matriz_path = "matriz_dummy_plantas.npy"
# Cargar matriz de sinergia
sinergia = np.load("matriz_sinergia.npy")

with open(grafo_path, "r") as f:
    grafo = json.load(f)

matriz_plantas = np.load(matriz_path)

n_nodos = len(grafo["nodes"])
n_tipos = matriz_plantas.shape[1]
edges = grafo["edges"]

# Construir lista de adyacencia para acceso rápido
vecinos = defaultdict(list)
for i, j in edges:
    vecinos[i].append(j)
    vecinos[j].append(i)

# Parámetros del algoritmo de hormigas (ACO completo)
n_hormigas = 50
n_iter = 200
alfa = 1.0      # importancia de feromona
beta = 2.0      # importancia de heurística
rho = 0.1       # tasa de evaporación
Q = 1.0         # constante para depositar feromona
elitismo = 5    # número de mejores soluciones que depositan feromona adicional
tau_min = 0.01  # feromona mínima
tau_max = 10.0  # feromona máxima

# Inicializar matriz de feromonas
feromonas = np.ones((n_nodos, n_tipos)) * 0.5

# Función para calcular heurística dinámica basada en vecinos ya asignados
def calcular_heuristica(nodo, tipo, asignacion_parcial, vecinos, sinergia):
    """Calcula la heurística para asignar un tipo a un nodo.
    
    La heurística considera la sinergia esperada con vecinos ya asignados.
    """
    valor_heuristica = 1.0
    vecinos_asignados = 0
    
    for vecino in vecinos[nodo]:
        # Si el vecino ya tiene tipo asignado
        tipo_vecino = np.argmax(asignacion_parcial[vecino])
        if asignacion_parcial[vecino].sum() > 0:  # vecino ya asignado
            # Sumar la sinergia (valores altos son mejores)
            valor_heuristica += sinergia[tipo, tipo_vecino]
            vecinos_asignados += 1
    
    # Normalizar por número de vecinos asignados
    if vecinos_asignados > 0:
        valor_heuristica = valor_heuristica / vecinos_asignados
    
    # Asegurar que sea positivo (sumar offset si hay valores negativos)
    return max(0.01, valor_heuristica)


# Función de costo usando matriz de sinergia
def costo(asignacion, edges, sinergia):
    """Calcula el costo total (negativo de la sinergia) de una asignación."""
    costo_total = 0.0
    for i, j in edges:
        tipo_i = np.argmax(asignacion[i])
        tipo_j = np.argmax(asignacion[j])
        # Queremos MAXIMIZAR sinergia, así que el costo es -sinergia
        costo_total -= sinergia[tipo_i, tipo_j]
    return costo_total


def busqueda_local_vecindad(asignacion, edges, vecinos, sinergia, max_intentos=200):
    """Búsqueda local optimizada: solo intercambia nodos conectados o cercanos."""
    mejor_asignacion = asignacion.copy()
    mejor_costo = costo(mejor_asignacion, edges, sinergia)
    
    # Convertir asignación a vector de tipos para acceso rápido
    tipos = np.argmax(mejor_asignacion, axis=1)
    
    intentos = 0
    mejoras_consecutivas = 0
    
    while intentos < max_intentos:
        intentos += 1
        
        # Seleccionar un nodo aleatorio
        nodo_i = np.random.randint(0, n_nodos)
        tipo_i = tipos[nodo_i]
        
        # Buscar un candidato entre vecinos o nodos aleatorios
        if len(vecinos[nodo_i]) > 0 and np.random.random() < 0.7:
            # 70% del tiempo: buscar entre vecinos directos
            nodo_j = np.random.choice(vecinos[nodo_i])
        else:
            # 30% del tiempo: buscar en un nodo aleatorio
            nodo_j = np.random.randint(0, n_nodos)
        
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
            delta_costo -= sinergia[tipo_i, tipo_vecino]  # quitar contribución vieja
            delta_costo += sinergia[tipo_j, tipo_vecino]  # agregar contribución nueva
        
        # Impacto de cambiar nodo_j de tipo_j a tipo_i
        for vecino in vecinos[nodo_j]:
            if vecino == nodo_i:  # Ya contado arriba
                continue
            tipo_vecino = tipos[vecino]
            delta_costo -= sinergia[tipo_j, tipo_vecino]
            delta_costo += sinergia[tipo_i, tipo_vecino]
        
        # Si mejora, aceptar el cambio
        if delta_costo < 0:  # Mejora (recordar que minimizamos -sinergia)
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

# Algoritmo principal ACO completo
mejor_asignacion = None
mejor_costo = float('inf')
historial_costos = []

print("Iniciando ACO completo...")
print(f"Parámetros: {n_hormigas} hormigas, {n_iter} iteraciones")
print(f"α={alfa}, β={beta}, ρ={rho}, Q={Q}, elitismo={elitismo}\n")

for iteracion in range(n_iter):
    soluciones = []
    costos_iter = []
    
    # Fase de construcción: cada hormiga construye una solución
    for h in range(n_hormigas):
        asignacion = np.zeros((n_nodos, n_tipos), dtype=int)
        plantas_restantes = matriz_plantas.sum(axis=0).copy()
        
        # Orden aleatorio de visita de nodos
        orden_nodos = np.random.permutation(n_nodos)
        
        for nodo_idx in orden_nodos:
            i = orden_nodos[nodo_idx]
            
            # Calcular heurística dinámica para cada tipo disponible
            heuristicas_tipos = np.zeros(n_tipos)
            for tipo in range(n_tipos):
                if plantas_restantes[tipo] > 0:
                    heuristicas_tipos[tipo] = calcular_heuristica(i, tipo, asignacion, vecinos, sinergia)
            
            # Probabilidad proporcional a feromona^alfa * heurística^beta
            probs = (feromonas[i] ** alfa) * (heuristicas_tipos ** beta)
            
            # Solo tipos con plantas disponibles
            probs = probs * (plantas_restantes > 0)
            
            if probs.sum() == 0:
                tipo = np.random.choice(np.where(plantas_restantes > 0)[0])
            else:
                probs = probs / probs.sum()
                tipo = np.random.choice(n_tipos, p=probs)
            
            asignacion[i, tipo] = 1
            plantas_restantes[tipo] -= 1
        
        # Búsqueda local para mejorar la solución
        asignacion_mejorada, costo_mejorado = busqueda_local_vecindad(asignacion, edges, vecinos, sinergia, max_intentos=200)
        
        soluciones.append(asignacion_mejorada)
        costos_iter.append(costo_mejorado)
    
    # Actualizar mejor solución global
    min_costo_iter = min(costos_iter)
    if min_costo_iter < mejor_costo:
        mejor_costo = min_costo_iter
        idx_mejor = costos_iter.index(min_costo_iter)
        mejor_asignacion = soluciones[idx_mejor].copy()
    
    historial_costos.append(mejor_costo)
    
    # Evaporación de feromonas
    feromonas = (1 - rho) * feromonas
    
    # Depositar feromona (solo las mejores hormigas)
    # Ordenar soluciones por costo
    indices_ordenados = np.argsort(costos_iter)
    
    for rank in range(min(elitismo, len(soluciones))):
        idx = indices_ordenados[rank]
        asignacion = soluciones[idx]
        costo_sol = costos_iter[idx]
        
        # Cantidad de feromona a depositar (inversamente proporcional al costo)
        delta_tau = Q / (1.0 + abs(costo_sol))
        
        # Depositar feromona en las asignaciones de esta solución
        for nodo in range(n_nodos):
            tipo = np.argmax(asignacion[nodo])
            feromonas[nodo, tipo] += delta_tau * (elitismo - rank)  # Más peso a mejores soluciones
    
    # Depositar feromona extra de la mejor solución global (elitismo global)
    if mejor_asignacion is not None:
        delta_tau_elite = Q / (1.0 + abs(mejor_costo))
        for nodo in range(n_nodos):
            tipo = np.argmax(mejor_asignacion[nodo])
            feromonas[nodo, tipo] += delta_tau_elite * elitismo * 2
    
    # Limitar feromonas entre tau_min y tau_max
    feromonas = np.clip(feromonas, tau_min, tau_max)
    
    # Mostrar progreso cada 10 iteraciones
    if (iteracion + 1) % 10 == 0:
        print(f"Iteración {iteracion+1}/{n_iter} - Mejor costo: {mejor_costo:.4f} - Costo promedio: {np.mean(costos_iter):.4f}")

print("\n" + "="*60)
print("OPTIMIZACIÓN COMPLETADA")
print("="*60)

# Guardar la mejor asignación encontrada
np.save("mejor_asignacion_hormigas.npy", mejor_asignacion)
print("\nMejor asignación guardada en 'mejor_asignacion_hormigas.npy'")
print(f"Costo mínimo encontrado: {mejor_costo:.4f}")
print(f"Sinergia total máxima: {-mejor_costo:.4f}")  # Negativo porque minimizamos -sinergia

# Guardar historial de costos
np.save("historial_costos_aco.npy", np.array(historial_costos))
print("Historial de costos guardado en 'historial_costos_aco.npy'")


