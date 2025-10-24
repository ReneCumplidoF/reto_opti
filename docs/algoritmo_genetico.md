# Algoritmo Genético - Optimización de Cantidades de Plantas

## 📋 Descripción General

Este algoritmo genético (GA) optimiza la **cantidad inicial de plantas por especie** (vector de 10 enteros) para minimizar la competencia total en el terreno. Utiliza el algoritmo ACO (`utils.hormigas.optimizar_aco`) como función de evaluación (fitness) para cada configuración de cantidades.

---

## 🎯 Objetivo

**Minimizar** la competencia total entre plantas vecinas encontrando la mejor distribución de cantidades por especie, respetando restricciones estrictas de viabilidad.

$$
\text{minimize} \quad f(\text{cromosoma}) = \text{competencia\_total\_ACO}(\text{cromosoma})
$$

---

## 🧬 Representación del Cromosoma

### Estructura
- **Tipo**: Vector de enteros de longitud 10
- **Dimensión**: Cada posición representa el conteo total de una especie
- **Orden**: `[Agave lechuguilla, Agave salmiana, Agave scabia, Agave striata, Opuntia cantabrigiensis, Opuntia engelmannii, Opuntia robusta, Opuntia streptacantha, Prosopis laevigata, Yucca filifera]`

### Conteos Base (referencia)
```python
BASE_COUNTS = [42, 196, 42, 42, 49, 38, 73, 64, 86, 26]
```

---

## 🔒 Restricciones

### 1. Límites por Gen (Especie)
Cada especie debe tener una cantidad dentro del **±10%** del conteo base:

$$
\text{gen}_i \in [\lfloor 0.9 \times \text{base}_i \rfloor, \lceil 1.1 \times \text{base}_i \rceil]
$$

**Ejemplo:**
- Agave salmiana (base = 196): rango válido = [176, 216]
- Opuntia robusta (base = 73): rango válido = [65, 81]

### 2. Suma Total Exacta
La suma de todos los genes debe ser **exactamente 658** (número de nodos en el grafo):

$$
\sum_{i=1}^{10} \text{gen}_i = 658
$$

### 3. Valores Enteros
Todos los genes deben ser números enteros positivos.

---

## ⚙️ Función de Fitness

### Evaluación
Para cada cromosoma:
1. Se llama a `optimizar_aco(plantas_totales=cromosoma, ...)`
2. ACO construye una asignación de plantas en el grafo usando esas cantidades
3. Se retorna el **mejor costo** (competencia mínima) encontrado por ACO

$$
\text{fitness}(\text{cromosoma}) = \min_{\text{asignación}} \sum_{(i,j) \in \text{aristas}} \text{competencia}[\text{tipo}_i, \text{tipo}_j]
$$

### Dirección de Optimización
- **Menor fitness = Mejor solución**
- Se busca minimizar la competencia total

### Caché de Evaluaciones
- Las evaluaciones se cachean por cromosoma único
- Evita re-ejecutar ACO para cromosomas ya evaluados
- Reduce drásticamente el tiempo de cómputo

---

## 🔄 Operadores Genéticos

### 1. Inicialización de Población
```python
def init_poblacion(n, rng):
    # Para cada individuo:
    #   1. Generar vector aleatorio en [LB, UB] por gen
    #   2. Reparar para cumplir suma = 658
    #   3. Agregar a la población
```

**Tamaño por defecto**: 40 individuos

### 2. Selección: Torneo
- **Método**: Torneo determinístico
- **Tamaño del torneo (k)**: 3 candidatos
- **Criterio**: Seleccionar el de menor fitness (mejor)

```python
def torneo_indices(fitness, k, rng):
    candidatos = random.sample(población, k)
    return min(candidatos, key=lambda x: fitness[x])
```

### 3. Cruza: Un Punto
- **Probabilidad**: 90% (p_cruza = 0.9)
- **Método**: Corte en un punto aleatorio

```python
def crossover_un_punto(p1, p2, rng):
    punto = random.randint(1, 9)  # Entre gen 1 y 9
    h1 = p1[:punto] + p2[punto:]
    h2 = p2[:punto] + p1[punto:]
    return h1, h2
```

**Nota**: Los hijos resultantes se reparan después para cumplir restricciones.

### 4. Mutación: Escalar por Gen
- **Probabilidad por gen**: 20% (p_mut = 0.2)
- **Paso máximo**: ±3 unidades
- **Método**: Para cada gen, con probabilidad p_mut:
  - Sumar delta aleatorio en [-3, +3] (sin incluir 0)
  - Si delta = 0, forzar ±1 aleatoriamente

```python
def mutacion_escalar(v, prob, rng, paso_max=3):
    for i in range(len(v)):
        if random() < prob:
            delta = randint(-paso_max, paso_max+1)
            if delta == 0:
                delta = 1 if random() < 0.5 else -1
            v[i] += delta
    return v
```

**Nota**: Después de mutar, se repara el vector para cumplir restricciones.

### 5. Reparación de Restricciones
Después de cruza y mutación, cada hijo se repara:

```python
def reparar_vector_sum_bounds(vec, suma_obj, lb, ub, rng):
    # 1. Clip a límites [LB, UB] por gen
    vec = np.clip(vec, lb, ub)
    
    # 2. Ajustar diferencia respecto a suma objetivo
    diff = suma_obj - vec.sum()
    
    # 3. Distribuir diff incrementando/decrementando genes
    #    sin violar límites, en orden aleatorio
    while diff != 0:
        shuffle(índices)
        for i in índices:
            if diff > 0 and vec[i] < ub[i]:
                vec[i] += 1
                diff -= 1
            elif diff < 0 and vec[i] > lb[i]:
                vec[i] -= 1
                diff += 1
    return vec
```

### 6. Elitismo
- **Cantidad**: 2 mejores individuos
- Los 2 mejores de cada generación pasan directamente a la siguiente
- Garantiza que la mejor solución nunca se pierda

---

## ⏱️ Criterio de Parada

### Tiempo Máximo
- **Por defecto**: 45 minutos (2700 segundos)
- **Comportamiento**: Al exceder el tiempo, se completa la generación actual y se detiene
- **Configurable**: Parámetro `tiempo_max_min` en `GAParams`

```python
while True:
    evaluar_poblacion()
    actualizar_mejor()
    
    if tiempo_transcurrido >= tiempo_max_min * 60:
        break  # Terminar tras cerrar generación
    
    generar_nueva_poblacion()
```

---

## 📊 Parámetros del Algoritmo

### Parámetros del GA

| Parámetro | Valor por Defecto | Descripción |
|-----------|-------------------|-------------|
| `pop_size` | 40 | Tamaño de la población |
| `torneo_k` | 3 | Candidatos en selección por torneo |
| `p_cruza` | 0.9 | Probabilidad de cruza (90%) |
| `p_mut` | 0.2 | Probabilidad de mutación por gen (20%) |
| `elitismo` | 2 | Número de mejores que pasan intactos |
| `tiempo_max_min` | 45.0 | Tiempo máximo de ejecución (minutos) |
| `seed` | None | Semilla aleatoria (None = aleatoria) |

### Parámetros de ACO (por evaluación)

| Parámetro | Valor por Defecto | Descripción |
|-----------|-------------------|-------------|
| `aco_hormigas` | 25 | Número de hormigas por iteración ACO |
| `aco_iter` | 25 | Número de iteraciones ACO |
| `aco_alfa` | 1.0 | Peso de la feromona (α) |
| `aco_beta` | 2.0 | Peso de la heurística (β) |
| `aco_rho` | 0.1 | Tasa de evaporación (ρ) |
| `aco_Q` | 1.0 | Constante de depósito de feromona |
| `aco_elitismo` | 5 | Mejores hormigas que depositan extra |
| `aco_max_intentos_busqueda` | 200 | Intentos en búsqueda local ACO |

**Nota**: Ajustar los parámetros ACO afecta el tiempo por evaluación y la precisión del fitness.

---

## 🚀 Uso

### Ejecución Básica (45 minutos)
```bash
cd /home/renec/clases/opti/reto_opti
python utils/genetico.py
```

### Ejecución con Parámetros Personalizados
```python
from utils.genetico import GAParams, ejecutar_ga

# Configurar parámetros
params = GAParams(
    pop_size=50,           # Población más grande
    tiempo_max_min=60.0,   # 1 hora
    aco_hormigas=30,       # Más hormigas por evaluación
    aco_iter=30,           # Más iteraciones ACO
    seed=42                # Semilla fija (reproducible)
)

# Ejecutar
resultado = ejecutar_ga(params)

# Acceder a resultados
print(f"Mejor cromosoma: {resultado['mejor_cromosoma']}")
print(f"Mejor costo: {resultado['mejor_costo']:.4f}")
```

### Ejecución Rápida (Prueba)
```python
params = GAParams(
    pop_size=10,
    tiempo_max_min=2.0,    # 2 minutos
    aco_hormigas=10,
    aco_iter=10,
    seed=123
)
ejecutar_ga(params)
```

---

## 📁 Archivos de Salida

### Generados por el GA

1. **`output/mejor_cromosoma_genetico.npy`**
   - Vector de 10 enteros con el mejor cromosoma encontrado
   - Formato: NumPy array shape (10,)
   
   ```python
   mejor = np.load('output/mejor_cromosoma_genetico.npy')
   # Ejemplo: [42, 191, 46, 40, 48, 40, 77, 57, 88, 29]
   ```

2. **`output/historial_costos_ga.npy`**
   - Mejor fitness (costo) por generación
   - Formato: NumPy array shape (n_generaciones,)
   - Útil para graficar convergencia

   ```python
   historial = np.load('output/historial_costos_ga.npy')
   plt.plot(historial)
   plt.xlabel('Generación')
   plt.ylabel('Mejor Costo (Competencia)')
   plt.title('Convergencia del GA')
   ```

3. **`output/resumen_ga.json`**
   - Resumen de la ejecución en formato JSON
   
   ```json
   {
     "mejor_costo": 892.7341,
     "mejor_cromosoma": [42, 191, 46, 40, 48, 40, 77, 57, 88, 29],
     "generaciones": 15,
     "tiempo_seg": 2700.45
   }
   ```

### Generados por ACO (para el mejor cromosoma)

4. **`output/mejor_asignacion_hormigas.npy`**
   - Matriz (658 × 10) con la asignación óptima de plantas en el grafo
   - Cada fila = nodo, cada columna = tipo de planta
   - Valor 1 en `[i, j]` indica que el nodo `i` tiene la planta tipo `j`

5. **`output/historial_costos_aco.npy`**
   - Historial de costos de la ejecución ACO final
   - Corresponde a la evaluación del mejor cromosoma con guardado activado

---

## 📈 Ejemplo de Salida en Consola

```
============================================================
INICIANDO OPTIMIZACIÓN ACO
============================================================
Parámetros: 25 hormigas, 25 iteraciones
α=1.0, β=2.0, ρ=0.1, Q=1.0, elitismo=5
τ_min=0.01, τ_max=10.0

OBJETIVO: MINIMIZAR competencia total entre vecinos
Nodos totales: 658 | Tipos de plantas: 10
Nodos fijos: 138 | Nodos libres: 520
Conteo objetivo por tipo: [42, 191, 46, 40, 48, 40, 77, 57, 88, 29]
Conteo restante a asignar: [35, 152, 38, 32, 39, 31, 63, 44, 61, 25]
============================================================

Iteración 10/25 | Mejor: 895.1234 | Promedio: 920.4567
Iteración 20/25 | Mejor: 892.7341 | Promedio: 910.2345

============================================================
OPTIMIZACIÓN COMPLETADA
============================================================
Competencia total mínima encontrada: 892.7341

✓ Mejor asignación → /path/to/output/mejor_asignacion_hormigas.npy
✓ Historial costos → /path/to/output/historial_costos_aco.npy

================ GA COMPLETADO ================
Tiempo total: 2700.45 s | Generaciones: 15
Mejor costo (ACO): 892.7341
Mejor cromosoma (10): [42, 191, 46, 40, 48, 40, 77, 57, 88, 29]
Guardado: 
- /path/to/output/mejor_cromosoma_genetico.npy
- /path/to/output/historial_costos_ga.npy
- /path/to/output/resumen_ga.json
```

---

## 🔬 Flujo del Algoritmo

```
┌─────────────────────────────────────────────────────────┐
│ 1. Inicialización                                       │
│    - Generar población aleatoria (40 individuos)        │
│    - Reparar restricciones (suma=658, límites)          │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ 2. Bucle Principal (hasta tiempo_max)                   │
│    ┌─────────────────────────────────────────────────┐  │
│    │ a. Evaluación de Fitness                        │  │
│    │    - Para cada individuo:                       │  │
│    │      * Llamar optimizar_aco(plantas_totales)    │  │
│    │      * Obtener mejor_costo como fitness         │  │
│    │    - Usar caché para evitar re-evaluaciones     │  │
│    └─────────────────────────────────────────────────┘  │
│    ┌─────────────────────────────────────────────────┐  │
│    │ b. Actualizar Mejor Global                      │  │
│    │    - Si hay mejor fitness: guardar cromosoma    │  │
│    │    - Registrar en historial                     │  │
│    └─────────────────────────────────────────────────┘  │
│    ┌─────────────────────────────────────────────────┐  │
│    │ c. Verificar Tiempo                             │  │
│    │    - Si tiempo >= tiempo_max: salir             │  │
│    └─────────────────────────────────────────────────┘  │
│    ┌─────────────────────────────────────────────────┐  │
│    │ d. Generar Nueva Población                      │  │
│    │    1. Elitismo: copiar 2 mejores                │  │
│    │    2. Reproducción (hasta llenar):              │  │
│    │       - Selección: torneo(k=3) x2               │  │
│    │       - Cruza: un punto (prob=0.9)              │  │
│    │       - Mutación: escalar (prob=0.2)            │  │
│    │       - Reparación: suma y límites              │  │
│    └─────────────────────────────────────────────────┘  │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ 3. Evaluación Final                                     │
│    - Ejecutar ACO completo sobre mejor cromosoma        │
│    - Guardar asignación y historial ACO                 │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ 4. Guardar Resultados                                   │
│    - mejor_cromosoma_genetico.npy                       │
│    - historial_costos_ga.npy                            │
│    - resumen_ga.json                                    │
│    - mejor_asignacion_hormigas.npy (desde ACO)          │
│    - historial_costos_aco.npy (desde ACO)               │
└─────────────────────────────────────────────────────────┘
```

---

## 🧪 Análisis de Resultados

### Cargar y Visualizar

```python
import numpy as np
import matplotlib.pyplot as plt

# Cargar mejor cromosoma
mejor = np.load('output/mejor_cromosoma_genetico.npy')
base = np.array([42, 196, 42, 42, 49, 38, 73, 64, 86, 26])

# Comparar con base
especies = ['A.lech', 'A.salm', 'A.scab', 'A.stri', 
            'O.cant', 'O.enge', 'O.robu', 'O.stre', 
            'P.laev', 'Y.fili']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico de barras comparativo
x = np.arange(10)
width = 0.35
ax1.bar(x - width/2, base, width, label='Base', alpha=0.8)
ax1.bar(x + width/2, mejor, width, label='Óptimo GA', alpha=0.8)
ax1.set_xticks(x)
ax1.set_xticklabels(especies, rotation=45, ha='right')
ax1.set_ylabel('Cantidad de plantas')
ax1.set_title('Comparación: Base vs Óptimo GA')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Diferencias porcentuales
diff_pct = ((mejor - base) / base) * 100
ax2.bar(especies, diff_pct, color=['green' if d >= 0 else 'red' for d in diff_pct])
ax2.axhline(0, color='black', linewidth=0.8)
ax2.set_ylabel('Diferencia (%)')
ax2.set_title('Cambio Porcentual respecto a Base')
ax2.set_xticklabels(especies, rotation=45, ha='right')
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('output/comparacion_cromosomas.png', dpi=150)
plt.show()

# Convergencia del GA
historial = np.load('output/historial_costos_ga.npy')
plt.figure(figsize=(10, 6))
plt.plot(historial, marker='o', linewidth=2)
plt.xlabel('Generación', fontsize=12)
plt.ylabel('Mejor Costo (Competencia)', fontsize=12)
plt.title('Convergencia del Algoritmo Genético', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('output/convergencia_ga.png', dpi=150)
plt.show()

# Validar restricciones
print("=" * 50)
print("VALIDACIÓN DE RESTRICCIONES")
print("=" * 50)
print(f"Suma total: {mejor.sum()} (objetivo: 658)")
print(f"Dentro de límites: {np.all((mejor >= np.floor(0.9*base)) & (mejor <= np.ceil(1.1*base)))}")
print("\nDetalles por especie:")
for i, esp in enumerate(especies):
    lb = int(np.floor(0.9 * base[i]))
    ub = int(np.ceil(1.1 * base[i]))
    print(f"  {esp:8s}: {mejor[i]:3d}  [válido: {lb}-{ub}]  ✓" if lb <= mejor[i] <= ub else "  ✗")
```

---

## 🎛️ Ajuste de Parámetros

### Balance Exploración-Explotación

| Para aumentar | Ajustar |
|--------------|---------|
| **Exploración** (diversidad) | ↑ `p_mut`, ↑ `pop_size`, ↓ `elitismo` |
| **Explotación** (intensificación) | ↓ `p_mut`, ↑ `elitismo`, ↑ `torneo_k` |

### Balance Precisión-Tiempo

| Objetivo | Configuración Sugerida |
|----------|------------------------|
| **Rápido** (exploración inicial) | `aco_hormigas=10`, `aco_iter=10`, `pop_size=20` |
| **Balanceado** (producción) | `aco_hormigas=25`, `aco_iter=25`, `pop_size=40` |
| **Preciso** (refinamiento) | `aco_hormigas=40`, `aco_iter=50`, `pop_size=60` |

---

## 💡 Consideraciones Técnicas

### Complejidad Computacional
- **Por generación**: O(pop_size × costo_ACO)
- **Costo ACO**: O(n_hormigas × n_iter × búsqueda_local)
- **Total**: Dominado por las evaluaciones ACO (>95% del tiempo)

### Caché de Evaluaciones
- Evita re-evaluar cromosomas idénticos
- Especialmente útil tras converger (muchos duplicados por elitismo)
- Típicamente reduce 20-40% de evaluaciones redundantes

### Reproducibilidad
- Fijar `seed` en `GAParams` garantiza resultados determinísticos
- Útil para experimentación y comparación

```python
# Mismos resultados en cada ejecución
params = GAParams(seed=42)
```

---

## 🐛 Solución de Problemas

### Error: "La suma debe ser exactamente 658"
- **Causa**: Bug en reparación o inicialización
- **Solución**: Verificar que `reparar_vector_sum_bounds` termine correctamente

### Evaluaciones muy lentas
- **Causa**: Parámetros ACO muy altos
- **Solución**: Reducir `aco_hormigas` y/o `aco_iter` (trade-off: precisión vs velocidad)

### No mejora tras muchas generaciones
- **Causa**: Convergencia prematura o población atrapada en óptimo local
- **Solución**: 
  - ↑ `p_mut` para más exploración
  - ↑ `pop_size` para más diversidad
  - Reiniciar con `seed` diferente

### Tiempo excedido pero sin resultados
- **Causa**: Primera generación no completó
- **Solución**: Reducir parámetros ACO o aumentar `tiempo_max_min`

---

## 📚 Referencias

### Algoritmos Genéticos
- Holland, J. H. (1992). *Adaptation in Natural and Artificial Systems*. MIT Press.
- Goldberg, D. E. (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning*. Addison-Wesley.

### ACO (función de fitness)
- Dorigo, M., & Stützle, T. (2004). *Ant Colony Optimization*. MIT Press.
- Ver: `docs/algoritmo_hormigas.md` (si existe) o `README.md`

### Optimización Multi-Objetivo con Restricciones
- Deb, K. (2001). *Multi-Objective Optimization using Evolutionary Algorithms*. Wiley.

---

## 📝 Notas Finales

- **Hibridación GA-ACO**: Este enfoque combina la exploración global del GA con la explotación local del ACO
- **Constraint Handling**: La reparación garantiza factibilidad sin penalizaciones en el fitness
- **Escalabilidad**: Tiempo por generación proporcional a `pop_size × aco_params`
- **Aplicación**: Optimizar configuraciones de entrada para algoritmos de asignación/scheduling

---

**Implementado en**: `utils/genetico.py`  
**Versión**: 1.0  
**Fecha**: Octubre 2025
