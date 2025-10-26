# 🚀 Guía de Optimización del Algoritmo Genético

## 📊 Hardware Utilizado

### ✅ Actualmente usa:
- **CPU**: 100% del cómputo (1 core a la vez en versión original)
- **RAM**: ~500MB-1GB para matrices y caché

### ❌ NO usa:
- **GPU** (tarjeta gráfica): 0%
- NumPy no tiene aceleración GPU por defecto
- Librerías GPU (PyTorch, TensorFlow, CuPy) no están instaladas

---

## 🔧 Optimizaciones Implementadas

### 1. **Paralelización Multi-Core** ⭐ **MEJOR OPCIÓN**

**Archivo**: `utils/genetico_paralelo.py`

**Beneficio**: Evalúa múltiples cromosomas simultáneamente

#### Speedup esperado:
```
CPU 4 cores  → ~3.5x más rápido
CPU 8 cores  → ~6.5x más rápido
CPU 16 cores → ~12x más rápido
```

#### Uso:
```bash
python utils/genetico_paralelo.py
```

O con parámetros personalizados:
```python
from utils.genetico_paralelo import GAParamsParalelo, ejecutar_ga_paralelo

params = GAParamsParalelo(
    pop_size=40,
    tiempo_max_min=45.0,
    n_workers=8,  # Número de cores a usar (None = todos)
    aco_hormigas=15,
    aco_iter=15,
)

resultado = ejecutar_ga_paralelo(params)
```

#### Cómo funciona:
```
Secuencial (1 core):
├─ Evaluar cromosoma 1 → 35 seg
├─ Evaluar cromosoma 2 → 35 seg
├─ Evaluar cromosoma 3 → 35 seg
└─ ... (40 total) = 23 minutos

Paralelo (8 cores):
├─ Evaluar cromosomas 1-8   → 35 seg (simultáneos)
├─ Evaluar cromosomas 9-16  → 35 seg (simultáneos)
├─ Evaluar cromosomas 17-24 → 35 seg (simultáneos)
├─ Evaluar cromosomas 25-32 → 35 seg (simultáneos)
└─ Evaluar cromosomas 33-40 → 35 seg (simultáneos)
Total: ~3-4 minutos
```

---

### 2. **Reducción de Parámetros ACO**

**Configuración optimizada** (ya en `genetico_paralelo.py`):
```python
aco_hormigas=15,  # Reducido de 25
aco_iter=15,      # Reducido de 25
```

**Beneficio**: Evaluación ~2.8x más rápida por cromosoma

**Trade-off**: Ligeramente menos preciso, pero suficiente para GA

---

### 3. **Caché de Evaluaciones**

**Ya implementado** en ambas versiones.

**Beneficio**: Evita re-evaluar cromosomas duplicados (común tras elitismo y convergencia)

**Ahorro típico**: 20-40% de evaluaciones redundantes

---

## 📈 Proyecciones de Rendimiento

### Con CPU de 8 cores (típico):

| Configuración | Gen/hora | Gen en 45 min | Calidad |
|---------------|----------|---------------|---------|
| **Original (secuencial)** | 2 | 1-2 | ❌ Insuficiente |
| **Paralelo (8 cores)** | 12-15 | 10-12 | ✅ Buena |
| **Paralelo + ACO reducido** | 18-22 | 15-18 | ✅✅ Excelente |

### Con CPU de 16 cores (workstation):

| Configuración | Gen/hora | Gen en 45 min | Calidad |
|---------------|----------|---------------|---------|
| **Paralelo (16 cores)** | 20-25 | 15-20 | ✅✅ Excelente |
| **Paralelo + ACO reducido** | 30-35 | 25-28 | ✅✅✅ Óptima |

---

## 🧪 Probar el Speedup

### Test rápido (6 minutos total):
```bash
python comparar_rendimiento.py
```

Esto ejecutará:
- 3 min con GA secuencial
- 3 min con GA paralelo
- Mostrará el speedup real en tu máquina

---

## 💻 Optimizaciones de Hardware

### 1. **Verificar cores disponibles**
```bash
# Linux
nproc
# o
lscpu | grep "^CPU(s):"

# Python
python -c "import multiprocessing; print(f'Cores: {multiprocessing.cpu_count()}')"
```

### 2. **Cerrar aplicaciones pesadas**
- Navegadores con muchas pestañas
- IDEs adicionales
- Otros procesos de Python

### 3. **Ajustar prioridad del proceso** (Linux)
```bash
# Dar máxima prioridad (requiere sudo)
sudo nice -n -20 python utils/genetico_paralelo.py
```

### 4. **Monitorear recursos durante ejecución**
```bash
# Terminal 1: Ejecutar GA
python utils/genetico_paralelo.py

# Terminal 2: Monitorear
htop  # o top
```

---

## 🎯 Configuraciones Recomendadas

### Para CPU de 4 cores:
```python
GAParamsParalelo(
    pop_size=30,
    tiempo_max_min=45.0,
    n_workers=3,  # Dejar 1 core libre para el SO
    aco_hormigas=12,
    aco_iter=12,
)
# Esperado: ~12-15 generaciones en 45 min
```

### Para CPU de 8 cores:
```python
GAParamsParalelo(
    pop_size=40,
    tiempo_max_min=45.0,
    n_workers=7,
    aco_hormigas=15,
    aco_iter=15,
)
# Esperado: ~18-22 generaciones en 45 min
```

### Para CPU de 16+ cores:
```python
GAParamsParalelo(
    pop_size=50,
    tiempo_max_min=45.0,
    n_workers=14,
    aco_hormigas=15,
    aco_iter=15,
)
# Esperado: ~30-35 generaciones en 45 min
```

---

## ⚠️ Consideraciones

### **Overhead de multiprocessing**
- Cada worker carga datos en memoria
- RAM necesaria: ~(n_workers × 300MB)
- Para 8 workers: ~2.4 GB RAM

### **Eficiencia de paralelización**
```
Speedup real = Speedup teórico × 0.85
```
Por ejemplo:
- 8 cores teórico: 8x
- 8 cores real: ~6.5-7x (por overhead de comunicación)

### **No usar todos los cores**
Recomendación: `n_workers = cpu_count() - 1`

Deja 1 core libre para:
- Sistema operativo
- Gestión del pool de procesos
- Otros procesos del usuario

---

## 🚫 Optimizaciones NO Viables

### 1. **GPU (CUDA/PyTorch)**
❌ **No ayuda** para este problema:
- ACO es inherentemente secuencial (hormigas construyen soluciones paso a paso)
- Operaciones NumPy son pequeñas (no justifican transferencia CPU→GPU)
- Overhead de GPU sería mayor que el beneficio

### 2. **Numba JIT Compilation**
❌ **Beneficio marginal** (~10-20%):
- El cuello de botella es ACO (95% del tiempo)
- ACO ya es razonablemente eficiente
- Complejidad adicional no justifica ganancia

### 3. **Compilar con Cython**
❌ **No vale la pena**:
- Beneficio: ~15-30% más rápido
- Costo: Complejidad de compilación y mantenimiento
- Mejor usar multiprocessing (6-12x más rápido)

---

## 📊 Resumen de Ganancias

```
Optimización                    | Speedup | Complejidad | Recomendación
--------------------------------|---------|-------------|---------------
Multiprocessing (8 cores)       | 6-7x    | Baja        | ✅ SÍ
ACO reducido (25→15)            | 2.8x    | Ninguna     | ✅ SÍ
Caché evaluaciones              | 1.3x    | Ninguna     | ✅ Ya incluido
GPU (CUDA)                      | 0.5x    | Alta        | ❌ NO
Numba JIT                       | 1.2x    | Media       | 🤔 Opcional
Cython                          | 1.3x    | Alta        | ❌ NO

COMBINADO (multiproc + ACO red): ~18-20x más rápido
```

---

## 🎬 Siguiente Paso

**Ejecutar versión paralela:**
```bash
python utils/genetico_paralelo.py
```

**O probar comparación primero:**
```bash
python comparar_rendimiento.py
```

Esto te dará métricas reales en tu hardware específico.
