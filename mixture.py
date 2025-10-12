from pyballistics import ozvb_lagrange, get_options_sample
import pandas as pd
import numpy as np
from pyballistics import get_powder_names
from tqdm import tqdm
import itertools

# Базовые настройки
opts = get_options_sample()
powder_names = get_powder_names()

# Параметры для перебора
k_values = [0.3, 0.5, 0.7]           # меньше значений для скорости
delta_values = [900, 1100]            # меньше значений для скорости
q = 5
d = opts['init_conditions']['d']

# Ограничения
MAX_PRESSURE = 390e6
MAX_BARREL_LENGTH = 65 * d

# Автоматическая генерация смесей
def generate_powder_mixtures(powders, max_powders_in_mixture=2):
    """Генерирует все возможные смеси порохов"""
    mixtures = []
    
    # Одиночные пороха
    for powder in powders:
        mixtures.append([{'name': powder, 'ratio': 1.0}])
    
    # Смеси из 2 порохов
    if max_powders_in_mixture >= 2:
        for powder1, powder2 in itertools.combinations(powders, 2):
            # Разные пропорции для смесей из 2 порохов
            ratios = [
                [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, 0.5],
                [0.6, 0.4], [0.7, 0.3], [0.8, 0.2]
            ]
            for ratio1, ratio2 in ratios:
                mixtures.append([
                    {'name': powder1, 'ratio': ratio1},
                    {'name': powder2, 'ratio': ratio2}
                ])
    
    # Смеси из 3 порохов (опционально)
    if max_powders_in_mixture >= 3:
        for powder1, powder2, powder3 in itertools.combinations(powders, 3):
            mixtures.append([
                {'name': powder1, 'ratio': 0.33},
                {'name': powder2, 'ratio': 0.33},
                {'name': powder3, 'ratio': 0.34}
            ])
    
    return mixtures

# Генерируем смеси (берем только первые 5 порохов для скорости)
test_powders = powder_names[:5]
powder_mixtures = generate_powder_mixtures(test_powders, max_powders_in_mixture=2)

print(f"Автоматическая генерация смесей...")
print(f"Исходные пороха: {test_powders}")
print(f"Сгенерировано смесей: {len(powder_mixtures)}")

# Создаем список для хранения результатов
results = []
total_iterations = len(powder_mixtures) * len(k_values) * len(delta_values)

print(f"\nНачало расчетов...")
print(f"Всего комбинаций: {total_iterations}")
print("=" * 60)

with tqdm(total=total_iterations, desc="Расчет смесей") as pbar:
    for mixture in powder_mixtures:
        mixture_name = " + ".join([f"{p['name']}({p['ratio']*100:.0f}%)" for p in mixture])
        
        for k in k_values:
            for delta in delta_values:
                # Вычисляем параметры
                omega_total = q * k
                W_0 = omega_total / delta
                
                # Распределяем массу по порохам согласно пропорциям
                powders_list = []
                for powder in mixture:
                    powder_omega = omega_total * powder['ratio']
                    powders_list.append({
                        'omega': powder_omega,
                        'dbname': powder['name']
                    })
                
                # Устанавливаем параметры
                opts['powders'] = powders_list
                opts['init_conditions']['W_0'] = W_0
                
                try:
                    result = ozvb_lagrange(opts)
                    
                    if 'layers' in result and len(result['layers']) > 0:
                        last_layer = result['layers'][-1]
                        final_velocity = last_layer['u'][-1]
                        max_pressure = max([max(layer['p']) for layer in result['layers']])
                        
                        # Получаем длину ствола
                        x_values = last_layer.get('x', np.array([]))
                        l_m = x_values[-1] if x_values.size > 0 else 0
                        
                        # Проверяем ограничения
                        constraints_ok = (max_pressure <= MAX_PRESSURE and 
                                        l_m <= MAX_BARREL_LENGTH)
                        
                        # Сохраняем результаты
                        results.append({
                            'mixture': mixture_name,
                            'mixture_composition': mixture,
                            'k': k,
                            'delta': delta,
                            'omega_total': omega_total,
                            'W_0': W_0,
                            'velocity': final_velocity,
                            'max_pressure': max_pressure,
                            'barrel_length': l_m,
                            'barrel_calibers': l_m / d,
                            'constraints_ok': constraints_ok,
                            'stop_reason': result.get('stop_reason', 'unknown'),
                            'success': True
                        })
                    else:
                        results.append({
                            'mixture': mixture_name,
                            'mixture_composition': mixture,
                            'k': k,
                            'delta': delta,
                            'omega_total': omega_total,
                            'W_0': W_0,
                            'velocity': 0,
                            'max_pressure': 0,
                            'barrel_length': 0,
                            'barrel_calibers': 0,
                            'constraints_ok': False,
                            'stop_reason': 'no_layers',
                            'success': False
                        })
                        
                except Exception as e:
                    results.append({
                        'mixture': mixture_name,
                        'mixture_composition': mixture,
                        'k': k,
                        'delta': delta,
                        'omega_total': omega_total,
                        'W_0': W_0,
                        'velocity': 0,
                        'max_pressure': 0,
                        'barrel_length': 0,
                        'barrel_calibers': 0,
                        'constraints_ok': False,
                        'stop_reason': f'error: {str(e)}',
                        'success': False
                    })
                
                pbar.update(1)

# Анализ результатов
df = pd.DataFrame(results)

print(f"\n" + "="*60)
print(f"ВСЕГО РАСЧЕТОВ: {len(df)}")
print(f"УСПЕШНЫХ: {df['success'].sum()}")
print(f"С ОГРАНИЧЕНИЯМИ: {df['constraints_ok'].sum()}")

# Лучшие результаты
good_results = df[(df['success']) & (df['constraints_ok'])]

if len(good_results) > 0:
    print(f"\n🎯 ТОП-10 ПО СКОРОСТИ:")
    top_velocity = good_results.nlargest(10, 'velocity')[[
        'mixture', 'k', 'delta', 'velocity', 'max_pressure', 'barrel_calibers'
    ]]
    
    for i, row in top_velocity.iterrows():
        print(f"{i+1:2d}. {row['mixture']:40} | "
              f"k={row['k']} | Δ={row['delta']} | "
              f"V={row['velocity']:6.1f} м/с | "
              f"P={row['max_pressure']/1e6:5.1f} МПа | "
              f"L={row['barrel_calibers']:4.1f} клб.")
    
    # Анализ лучших смесей
    print(f"\n📊 СТАТИСТИКА ЛУЧШИХ СМЕСЕЙ:")
    mixture_stats = good_results.groupby('mixture').agg({
        'velocity': ['count', 'mean', 'max'],
        'max_pressure': 'mean'
    }).round(2)
    
    # Сортируем по максимальной скорости
    mixture_stats = mixture_stats.sort_values(('velocity', 'max'), ascending=False)
    print(mixture_stats.head(10))
    
else:
    print("\n❌ Нет результатов, удовлетворяющих ограничениям")

# Анализ типов смесей
print(f"\n🔬 АНАЛИЗ ТИПОВ СМЕСЕЙ:")
single_powder = df[df['mixture'].str.contains(r'\(100\%\)')]
mixed_powders = df[~df['mixture'].str.contains(r'\(100\%\)')]

print(f"Одиночные пороха: {len(single_powder[single_powder['success']])} успешных")
print(f"Смеси порохов: {len(mixed_powders[mixed_powders['success']])} успешных")

if len(single_powder[single_powder['success']]) > 0 and len(mixed_powders[mixed_powders['success']]) > 0:
    single_success = single_powder[single_powder['success']]
    mixed_success = mixed_powders[mixed_powders['success']]
    
    print(f"Макс. скорость (одиночные): {single_success['velocity'].max():.1f} м/с")
    print(f"Макс. скорость (смеси): {mixed_success['velocity'].max():.1f} м/с")