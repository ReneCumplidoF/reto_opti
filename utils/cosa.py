import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick # Para formatear el eje Y como porcentaje

# 1. Definir los datos de la imagen
data = {
    'Especies': [
        'Agave lechuguilla', 'Agave salmiana', 'Agave scabra', 'Agave striata',
        'Opuntia cantabrigiensis', 'Opuntia engelmannii', 'Opuntia robusta',
        'Opuntia streptacantha', 'Prosopis laevigata', 'Yucca filifera'
    ],
    'Cantidad': [
        42, 196, 42, 42, 49, 38, 73, 64, 86, 26
    ]
}

# 2. Crear un DataFrame de pandas
df = pd.DataFrame(data)

# 3. Calcular el total y la probabilidad
total_plantas = df['Cantidad'].sum() # Esto debe dar 658
df['Probabilidad'] = df['Cantidad'] / total_plantas

# 4. Ordenar el DataFrame por probabilidad (opcional, pero hace el gráfico más legible)
df_sorted = df.sort_values(by='Probabilidad', ascending=False)

# 5. Crear el gráfico
plt.figure(figsize=(12, 7)) # Ajustar el tamaño para que quepan las etiquetas

# Crear las barras
bars = plt.bar(df_sorted['Especies'], df_sorted['Probabilidad'], color='skyblue')

# Añadir título y etiquetas
plt.title('Distribución de Probabilidad de Especies por Ha.', fontsize=16)
plt.xlabel('Especies', fontsize=12)
plt.ylabel('Proporción del Total (Probabilidad)', fontsize=12)

# Rotar las etiquetas del eje X para que no se solapen
plt.xticks(rotation=45, ha='right')

# Formatear el eje Y para mostrar porcentajes
# gca() = "get current axis" (obtener el eje actual)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

# Añadir etiquetas de porcentaje encima de cada barra
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f'{yval*100:.1f}%', ha='center', va='bottom')

# Ajustar el diseño para que todo quepa bien
plt.tight_layout()

# 6. Mostrar el gráfico
plt.show()
