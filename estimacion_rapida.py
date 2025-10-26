"""
Estimación rápida del tiempo por evaluación ACO.
Usa parámetros pequeños para medir y luego extrapola.
"""
from utils.genetico import GAParams, ejecutar_ga
import time
import numpy as np

print("="*60)
print("ESTIMACIÓN RÁPIDA DE TIEMPO")
print("="*60)

# Paso 1: Medir tiempo de 1 evaluación pequeña
print("\n1. Midiendo tiempo de evaluación ACO pequeña...")
params_mini = GAParams(
    pop_size=3,           # Solo 3 individuos
    tiempo_max_min=0.05,  # 3 segundos (completa 1 gen)
    aco_hormigas=5,       # ACO muy pequeño
    aco_iter=5,
    elitismo=1,
    seed=123
)

t0 = time.time()
resultado_mini = ejecutar_ga(params_mini)
t_mini = time.time() - t0

eval_por_gen_mini = params_mini.pop_size  # Primera gen evalúa todo
t_por_eval_mini = t_mini / eval_por_gen_mini

print(f"   ✓ Completado: {resultado_mini['generaciones']} generación(es)")
print(f"   ✓ Tiempo total: {t_mini:.2f} seg")
print(f"   ✓ Tiempo por evaluación (ACO 5x5): {t_por_eval_mini:.2f} seg")

# Paso 2: Extrapolar a parámetros de producción
print("\n2. Extrapolando a parámetros de producción...")

# ACO escala aproximadamente lineal con (hormigas × iteraciones)
factor_aco = (25 * 25) / (5 * 5)  # 25x25 vs 5x5
t_por_eval_prod = t_por_eval_mini * factor_aco

print(f"   ✓ Factor de escalado ACO: {factor_aco:.1f}x")
print(f"   ✓ Tiempo estimado por evaluación (ACO 25x25): {t_por_eval_prod:.2f} seg")

# Paso 3: Calcular generaciones para diferentes tiempos
pop_size = 40
print(f"\n3. Estimaciones para población de {pop_size} individuos:")
print("-"*60)

# Primera generación (sin caché)
t_gen_1 = pop_size * t_por_eval_prod
print(f"   Primera generación (sin caché): {t_gen_1:.1f} seg ({t_gen_1/60:.2f} min)")

# Generaciones siguientes (con caché ~20-30% y elitismo)
evals_siguientes = pop_size * 0.75  # ~75% son nuevas evaluaciones
t_gen_siguiente = evals_siguientes * t_por_eval_prod
print(f"   Generaciones 2+  (con caché):   {t_gen_siguiente:.1f} seg ({t_gen_siguiente/60:.2f} min)")

print("\n4. Proyección para diferentes tiempos límite:")
print("-"*60)

tiempos_limite = [5, 15, 30, 45, 60]
for tiempo_min in tiempos_limite:
    tiempo_seg = tiempo_min * 60
    # Primera gen + N generaciones siguientes
    gens = 1 + max(0, (tiempo_seg - t_gen_1) / t_gen_siguiente)
    print(f"   {tiempo_min:3d} min → ~{int(gens):3d} generaciones")

print("\n5. Recomendación:")
print("-"*60)
# Para 45 minutos
tiempo_objetivo = 45 * 60
gens_45min = 1 + max(0, (tiempo_objetivo - t_gen_1) / t_gen_siguiente)
print(f"   Con tiempo_max_min=45.0:")
print(f"   - Generaciones esperadas: ~{int(gens_45min)}")
print(f"   - Mejor cromosoma encontrado al final")
print(f"   - Convergencia típica: 15-25 generaciones")

print("\n" + "="*60)
print("NOTA: Estos son estimados. El tiempo real puede variar ±30%")
print("      según la complejidad del grafo y el hardware.")
print("="*60)
