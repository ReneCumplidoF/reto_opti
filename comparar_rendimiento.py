"""
Compara rendimiento: GA secuencial vs GA paralelo
"""
import subprocess
import time

print("="*70)
print("COMPARACIÓN: GA SECUENCIAL vs GA PARALELO")
print("="*70)

# Configuración de prueba rápida (5 min)
config = """
from utils.genetico import GAParams, ejecutar_ga
from utils.genetico_paralelo import GAParamsParalelo, ejecutar_ga_paralelo
import time

params_seq = GAParams(
    pop_size=20,
    tiempo_max_min=3.0,
    aco_hormigas=10,
    aco_iter=10,
    seed=123
)

params_par = GAParamsParalelo(
    pop_size=20,
    tiempo_max_min=3.0,
    n_workers=None,  # Todos los cores
    aco_hormigas=10,
    aco_iter=10,
    seed=123
)

print("\\n" + "="*70)
print("TEST 1: GA SECUENCIAL")
print("="*70)
t0 = time.time()
r1 = ejecutar_ga(params_seq)
t_seq = time.time() - t0

print("\\n" + "="*70)
print("TEST 2: GA PARALELO")
print("="*70)
t0 = time.time()
r2 = ejecutar_ga_paralelo(params_par)
t_par = time.time() - t0

print("\\n" + "="*70)
print("RESULTADOS COMPARATIVOS")
print("="*70)
print(f"Secuencial:")
print(f"  - Tiempo: {t_seq:.1f}s")
print(f"  - Generaciones: {r1['generaciones']}")
print(f"  - Tiempo/gen: {t_seq/r1['generaciones']:.1f}s")
print()
print(f"Paralelo:")
print(f"  - Tiempo: {t_par:.1f}s")
print(f"  - Generaciones: {r2['generaciones']}")
print(f"  - Tiempo/gen: {t_par/r2['generaciones']:.1f}s")
print()
print(f"🚀 Speedup: {t_seq/t_par:.2f}x más rápido")
print(f"📈 Generaciones extra: +{r2['generaciones'] - r1['generaciones']}")
print("="*70)
"""

with open('/tmp/test_comparacion_ga.py', 'w') as f:
    f.write(config)

print("\n⏳ Ejecutando prueba de 3 minutos por cada versión...")
print("   (total: ~6 minutos)\n")

subprocess.run([
    "/home/renec/clases/opti/reto_opti/.venv/bin/python",
    "/tmp/test_comparacion_ga.py"
])
