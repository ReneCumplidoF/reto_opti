import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd()))
from utils.paths import data_path

with open(data_path('hectarea.json'), 'r') as f:
    data = json.load(f)

conteos = data['conteos_por_especie']
especies = data['especies']
total = data['total_plantas']

print('='*70)
print('DISTRIBUCIÓN INICIAL DE PLANTAS (1 Hectárea Fija)')
print('='*70)
print(f'\nTotal de plantas pre-asignadas: {total}')
print(f'Nodos libres restantes: {658 - total} (de 658 total)\n')
print(f"{'Especie':<35} {'Cantidad':<12} {'Porcentaje'}")
print('-'*70)

for i, (nombre, cant) in enumerate(zip(especies, conteos.values())):
    print(f'E{i+1} - {nombre:<30} {cant:<12} {100*cant/total:.1f}%')

print('-'*70)
print(f'\nConteos: {list(conteos.values())}')
print('='*70)
