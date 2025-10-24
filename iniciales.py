import pandas as pd
import numpy as np

def simulacion_montecarlo(df_densidades, especie, area_objetivo_ha, num_simulaciones=10000):
    """
    Realiza una simulación de Montecarlo para una especie y un área dada.
    """
    # 1. Obtener la distribución de densidades observadas de la tabla
    # Esto es un array de NumPy con los 30 valores de densidad para esa especie
    distribucion_densidad = df_densidades.loc[especie].values
    
    # 2. Muestreo de Montecarlo (Bootstrap)
    # Elegimos aleatoriamente (con reemplazo) de nuestra distribución 
    # de densidades, N veces (num_simulaciones).
    densidades_simuladas = np.random.choice(
        distribucion_densidad, 
        size=num_simulaciones, 
        replace=True
    )
    
    # 3. Calcular los conteos simulados para el área objetivo
    # Multiplicamos cada densidad simulada por el tamaño del nuevo terreno
    conteos_simulados = densidades_simuladas * area_objetivo_ha
    
    # 4. Calcular estadísticas
    valor_esperado = np.mean(conteos_simulados)
    mediana = np.median(conteos_simulados)
    # Intervalo de confianza del 95%
    ci_low = np.percentile(conteos_simulados, 2.5)
    ci_high = np.percentile(conteos_simulados, 97.5)
    
    return {
        "especie": especie,
        "valor_esperado": valor_esperado,
        "mediana": mediana,
        "intervalo_95_conf": (ci_low, ci_high)
    }

# --- 1. PREPARACIÓN DE DATOS ---
# Transcripción de los datos de tu imagen
data = {
    'Especies': [
        'Agave lechuguilla', 'Agave salmiana', 'Agave scabia', 'Agave striata', 
        'Opuntia cantabrigiensis', 'Opuntia engelmannii', 'Opuntia robusta', 
        'Opuntia streptacantha', 'Prosopis laevigata', 'Yucca filifera', 
        'Área del poligono (Ha)'
    ],
    'P1': [8, 46, 16, 16, 11, 14, 18, 12, 15, 9, 1.28],
    'P2': [58, 263, 47, 49, 60, 44, 111, 95, 98, 39, 6.64],
    'P3': [66, 236, 51, 50, 71, 56, 100, 78, 123, 28, 6.76],
    'P4': [10, 52, 15, 9, 15, 10, 18, 20, 23, 11, 1.38],
    'P5': [65, 280, 66, 61, 92, 54, 124, 114, 106, 41, 8.0],
    'P6': [67, 306, 63, 61, 81, 62, 118, 91, 133, 45, 7.82],
    'P7': [41, 209, 43, 49, 48, 52, 87, 60, 97, 27, 5.53],
    'P8': [32, 252, 46, 40, 73, 39, 86, 79, 91, 26, 5.64],
    'P9': [58, 269, 58, 53, 57, 58, 94, 91, 117, 37, 7.11],
    'P10': [47, 209, 47, 41, 66, 43, 96, 71, 108, 29, 6.11],
    'P11': [48, 233, 41, 37, 48, 40, 94, 65, 94, 27, 5.64],
    'P12': [36, 182, 43, 44, 51, 41, 68, 72, 67, 19, 4.92],
    'P13': [50, 189, 49, 28, 43, 33, 73, 76, 97, 25, 5.05],
    'P14': [35, 186, 38, 43, 39, 44, 73, 61, 74, 23, 4.75],
    'P15': [40, 311, 52, 62, 68, 63, 132, 89, 149, 53, 7.97],
    'P16': [70, 290, 55, 54, 84, 50, 105, 77, 121, 40, 7.34],
    'P17': [56, 233, 39, 46, 71, 45, 83, 81, 97, 32, 5.98],
    'P18': [45, 204, 34, 33, 56, 46, 82, 48, 95, 17, 5.4],
    'P19': [56, 223, 51, 48, 50, 48, 93, 90, 120, 33, 6.28],
    'P20': [69, 285, 57, 75, 77, 74, 107, 103, 104, 25, 7.6],
    'P21': [54, 287, 60, 62, 89, 58, 126, 109, 139, 44, 8.0],
    'P22': [75, 292, 69, 68, 104, 62, 109, 95, 125, 43, 8.0],
    'P23': [44, 288, 15, 51, 69, 60, 122, 100, 106, 39, 7.87],
    'P24': [10, 59, 28, 11, 17, 18, 17, 10, 39, 8, 1.47],
    'P25': [44, 155, 73, 31, 39, 39, 65, 50, 67, 25, 4.19],
    'P26': [72, 291, 52, 50, 66, 62, 125, 101, 122, 31, 7.52],
    'P27': [74, 342, 67, 55, 70, 76, 100, 100, 129, 44, 8.0],
    'P28': [67, 309, 61, 66, 86, 54, 120, 103, 141, 33, 7.56],
    'P29': [60, 280, 56, 53, 76, 68, 109, 85, 135, 36, 5.6],
    'P30': [34, 195, 56, 58, 55, 46, 91, 63, 86, 23, 5.4],
}

df = pd.DataFrame(data)
df = df.set_index('Especies')

# --- 2. CALCULAR DENSIDADES ---
# Separar las áreas del resto de los conteos
df_areas = df.loc['Área del poligono (Ha)']
df_counts = df.drop('Área del poligono (Ha)')

df_densidades = df_counts.div(df_areas, axis='columns')

""" 
print("--- Densidades Calculadas (Plantas/Ha) ---")
print(df_densidades.iloc[:, :5])
print("-" * 40) """

# -------- ¡EDITA ESTOS VALORES! --------
TERRENO_OBJETIVO_HA = 1  # ¿De cuántas hectáreas es el terreno que quieres simular?
NUM_SIMULACIONES = 10000   # Número de repeticiones (más es más preciso)
# ----------------------------------------

print(f"--- Simulación de Montecarlo para un terreno de {TERRENO_OBJETIVO_HA} Ha ---")
print(f"(Basado en {NUM_SIMULACIONES} simulaciones por especie)\n")

resultados_finales = []
# Iterar sobre cada especie y correr la simulación
for especie in df_densidades.index:
    resultado = simulacion_montecarlo(
        df_densidades, 
        especie, 
        TERRENO_OBJETIVO_HA, 
        NUM_SIMULACIONES
    )
    resultados_finales.append(resultado)

#cantidad de plantas requeridas por especie

requeridas = [42,196,42,42,49,38,73,64,86,26,658]
totales = []

# Imprimir los resultados
for i in range(len(resultados_finales)):
    res = resultados_finales[i]
    valor_esperado = res['valor_esperado']
    totales.append(requeridas[i]-valor_esperado)
    #print(f"[{res['especie']}]", f"  = {res['valor_esperado']:.2f} plantas", f"  total: {totales[i]:.2f} plantas" )



# totales es la variable que contiene la cantidad de plantas faltantes por especie (lo que debe comprarse)