# Reto de Optimización - Asignación de Plantas

Este proyecto implementa un algoritmo ACO (Ant Colony Optimization) para optimizar la asignación de diferentes tipos de plantas en una malla hexagonal, **minimizando la competencia** entre plantas vecinas.

## 📁 Estructura del Proyecto

```
reto_opti/
├── datos/                      # Datos de entrada
│   ├── grafo_hexagonal.json   # Grafo de la malla hexagonal
│   ├── info_actual.csv         # Información de especies y polígonos
│   ├── hectarea.json           # Asignación inicial de 1 Ha
│   ├── matriz_competencia.npy # Matriz de competencia entre especies
│   └── ...
├── output/                     # Resultados generados (git-ignored)
│   ├── mejor_asignacion_hormigas.npy
│   ├── historial_costos_aco.npy
│   └── grafo_hexagonal.png
├── utils/                      # Utilidades y scripts auxiliares
│   ├── __init__.py
│   ├── paths.py               # Funciones para manejo de rutas
│   ├── matriz_competencia.py # Generación de matriz de competencia
│   ├── generacion_grafo.py   # Generación del grafo hexagonal
│   ├── generar_hectarea.py   # Generación de asignación inicial
│   └── ...
├── algoritmo_hormigas.ipynb   # Notebook principal con ACO
└── requirements.txt           # Dependencias Python

```

## 🎯 Objetivo

**Minimizar la competencia total** entre plantas vecinas en la malla:

$$\text{minimize} \quad \sum_{(i,j) \in \text{aristas}} \text{competencia}[tipo_i, tipo_j]$$

Donde:
- `competencia[i, j]` ∈ [0, 1] representa el nivel de competencia entre especies
- Valores altos = alta competencia = mala compatibilidad
- El algoritmo busca configuraciones con mínima competencia total

## 🚀 Uso Rápido

### 1. Instalar dependencias

```bash
make install
```

### 2. Generar archivos base (si es necesario)

```bash
# Generar matriz de competencia
python3 utils/matriz_competencia.py

# Generar grafo hexagonal
python3 utils/generacion_grafo.py

# Generar asignación inicial de 1 Ha
python3 utils/generar_hectarea.py
```

### 3. Ejecutar optimización ACO

Abrir `algoritmo_hormigas.ipynb` y ejecutar todas las celdas.

## 📊 Archivos de Entrada

### `datos/matriz_competencia.npy`
Matriz 10×10 con valores de competencia entre especies:
- Diagonal = 1.0 (máxima competencia consigo misma)
- Valores altos = alta competencia (evitar)
- Valores bajos = baja competencia (preferir)

### `datos/grafo_hexagonal.json`
Estructura del grafo:
```json
{
  "nodes": [{"id": 0, "x": 1.23, "y": 4.56}, ...],
  "edges": [[0, 1], [0, 2], ...]
}
```

### `datos/info_actual.csv`
Conteos observados de especies en 30 polígonos para calcular densidades y proporciones.

## 📈 Archivos de Salida

### `output/mejor_asignacion_hormigas.npy`
Matriz (658 × 10) con la mejor asignación encontrada:
- Cada fila = un nodo (posición en la malla)
- Cada columna = un tipo de planta
- Valor 1 en `[i, j]` indica que el nodo `i` tiene la planta tipo `j`

### `output/historial_costos_aco.npy`
Vector con el mejor costo (competencia mínima) en cada iteración del algoritmo.

## 🔧 Utilidades

### `utils/paths.py`
Funciones helper para manejo consistente de rutas:
```python
from utils.paths import data_path, output_path

# Leer desde datos/
with open(data_path("grafo_hexagonal.json")) as f:
    grafo = json.load(f)

# Guardar en output/
np.save(output_path("resultado.npy"), array)
```

## 🐜 Algoritmo ACO

El algoritmo implementa:
- **Construcción probabilística** usando feromonas y heurística inversa a competencia
- **Búsqueda local optimizada** con cálculo incremental de costos
- **Elitismo dual** (mejores por iteración + mejor histórica)
- **Evaporación de feromonas** para evitar convergencia prematura

Ver documentación completa en el notebook `algoritmo_hormigas.ipynb`.

## 📝 Notas Importantes

1. **Cambio de paradigma**: El proyecto ahora minimiza **competencia** (antes maximizaba sinergia)
2. **Estructura de carpetas**: 
   - `datos/` para entradas
   - `output/` para resultados (git-ignored)
   - `utils/` para código auxiliar
3. **Matriz de competencia**: Valores altos son malos, valores bajos son buenos

## 🛠️ Comandos Make Disponibles

```bash
make help          # Ver todos los comandos
make install       # Instalar dependencias
make notebook      # Abrir Jupyter Lab
make clean-cache   # Limpiar archivos temporales
```

## 📚 Referencias

- Dorigo, M., & Stützle, T. (2004). *Ant Colony Optimization*. MIT Press.
- [Documentación completa en algoritmo_hormigas.ipynb]
