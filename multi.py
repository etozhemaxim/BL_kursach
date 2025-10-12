from pyballistics import ozvb_lagrange, get_options_sample, ozvb_termo
import pandas as pd
import numpy as np
from pyballistics import get_powder_names
import copy
from tqdm import tqdm
import concurrent.futures
import multiprocessing as mp
from functools import partial

# Глобальные константы
opts = get_options_sample()
powder_names = get_powder_names()

# Параметры для перебора
k_values = np.arange(0.01, 1.01, 0.2)
delta_values = np.arange(800, 1201, 100)
q = 5  # масса снаряда
d = opts['init_conditions']['d']  # калибр

# Ограничения
MAX_PRESSURE = 390e6  # 390 МПа
MAX_BARREL_LENGTH = 65 * d  # 65 калибров
MAX_MUZZLE_PRESSURE_P50 = 180e6  # 180 МПа при +50°C
MAX_VELOCITY_M50 = 830  # 830 м/с при -50°C

def calculate_single_case(params):
    """Функция для расчета одного случая - будет выполняться в отдельном потоке"""
    powder_name, k, delta = params
    
    # Создаем копию opts для каждого потока
    local_opts = copy.deepcopy(opts)
    
    # Вычисляем параметры
    omega = q * k
    W_0 = omega / delta
    
    # Устанавливаем параметры
    local_opts['powders'] = [{'omega': omega, 'dbname': powder_name}]
    local_opts['init_conditions']['W_0'] = W_0
    
    try:
        result = ozvb_lagrange(local_opts)
        
        if 'layers' in result and len(result['layers']) > 0:
            last_layer = result['layers'][-1]
            final_velocity = last_layer['u'][-1]
            max_pressure = max([max(layer['p']) for layer in result['layers']])
            
            # Получаем длину ствола из последнего слоя
            x_values = last_layer.get('x', np.array([]))
            if x_values.size > 0:
                l_m = x_values[-1]  # координата снаряда = длина ствола
            else:
                l_m = 0
            
            # ПРОВЕРКА ОГРАНИЧЕНИЙ
            constraints_violated = []
            constraints_passed = True
            
            # 1. Проверка максимального давления
            if max_pressure > MAX_PRESSURE:
                constraints_violated.append(f'pressure_{max_pressure/1e6:.1f}MPa')
                constraints_passed = False
            
            # 2. Проверка максимальной длины ствола
            if l_m > MAX_BARREL_LENGTH:
                constraints_violated.append(f'barrel_length_{l_m/d:.1f}calibers')
                constraints_passed = False
            
            # 3. Проверка дульной скорости при -50°C
            velocity_m50 = 0
            try:
                # Создаем копию опций для расчета при -50°C
                opts_m50 = copy.deepcopy(local_opts)
                opts_m50['init_conditions']['T_0'] = 223.15  # -50°C в Кельвинах
                opts_m50['stop_conditions'] = {'v_p': MAX_VELOCITY_M50 + 100, 'steps_max': 100000}
                
                result_m50 = ozvb_termo(opts_m50)
                velocity_m50 = result_m50.get('v_p', [0])[-1]
                
                if velocity_m50 > MAX_VELOCITY_M50:
                    constraints_violated.append(f'velocity_m50_{velocity_m50:.1f}m/s')
                    constraints_passed = False
            except Exception as e:
                constraints_violated.append(f'velocity_m50_error: {str(e)}')
                constraints_passed = False
            
            # 4. Проверка давления на дульном срезе при +50°C
            p_mz_p50 = 0
            try:
                # Создаем копию опций для расчета при +50°C
                opts_p50 = copy.deepcopy(local_opts)
                opts_p50['init_conditions']['T_0'] = 323.15  # +50°C в Кельвинах
                opts_p50['stop_conditions'] = {'p_max': MAX_MUZZLE_PRESSURE_P50 * 2, 'steps_max': 100000}
                
                result_p50 = ozvb_termo(opts_p50)
                p_mz_p50 = result_p50.get('p_m', [0])[-1] if 'p_m' in result_p50 else 0
                
                if p_mz_p50 > MAX_MUZZLE_PRESSURE_P50:
                    constraints_violated.append(f'muzzle_pressure_p50_{p_mz_p50/1e6:.1f}MPa')
                    constraints_passed = False
            except Exception as e:
                constraints_violated.append(f'muzzle_pressure_p50_error: {str(e)}')
                constraints_passed = False
            
            return {
                'powder': powder_name,
                'k': k,
                'delta': delta,
                'omega': omega,
                'W_0': W_0,
                'velocity': final_velocity,
                'max_pressure': max_pressure,
                'barrel_length': l_m,
                'barrel_length_calibers': l_m / d if d > 0 else 0,
                'velocity_m50': velocity_m50,
                'muzzle_pressure_p50': p_mz_p50,
                'stop_reason': result.get('stop_reason', 'unknown'),
                'constraints_passed': constraints_passed,
                'constraints_violated': ', '.join(constraints_violated) if constraints_violated else 'all_passed',
                'success': True
            }
        else:
            return {
                'powder': powder_name,
                'k': k,
                'delta': delta,
                'omega': omega,
                'W_0': W_0,
                'velocity': 0,
                'max_pressure': 0,
                'barrel_length': 0,
                'barrel_length_calibers': 0,
                'velocity_m50': 0,
                'muzzle_pressure_p50': 0,
                'stop_reason': 'no_layers',
                'constraints_passed': False,
                'constraints_violated': 'no_layers_data',
                'success': False
            }
            
    except Exception as e:
        return {
            'powder': powder_name,
            'k': k,
            'delta': delta,
            'omega': omega,
            'W_0': W_0,
            'velocity': 0,
            'max_pressure': 0,
            'barrel_length': 0,
            'barrel_length_calibers': 0,
            'velocity_m50': 0,
            'muzzle_pressure_p50': 0,
            'stop_reason': f'error: {str(e)}',
            'constraints_passed': False,
            'constraints_violated': f'calculation_error: {str(e)}',
            'success': False
        }

def main():
    # Создаем список всех комбинаций параметров
    all_combinations = []
    for powder_name in powder_names[:10]:
        for k in k_values:
            for delta in delta_values:
                all_combinations.append((powder_name, k, delta))
    
    total_iterations = len(all_combinations)
    
    print(f"Начало многопоточных расчетов...")
    print(f"Всего итераций: {total_iterations}")
    print(f"Порохов: {len(powder_names[:1])}")
    print(f"Коэффициентов k: {len(k_values)}")
    print(f"Плотностей Δ: {len(delta_values)}")
    print(f"Количество ядер CPU: {mp.cpu_count()}")
    print("=" * 60)
    
    results = []
    
    # Используем ThreadPoolExecutor для многопоточности
    # Если расчеты CPU-bound, лучше использовать ProcessPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
        # Запускаем все задачи и создаем прогресс-бар
        with tqdm(total=total_iterations, desc="Выполнение расчетов") as pbar:
            # Запускаем все задачи асинхронно
            future_to_params = {
                executor.submit(calculate_single_case, params): params 
                for params in all_combinations
            }
            
            # Обрабатываем результаты по мере их поступления
            for future in concurrent.futures.as_completed(future_to_params):
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Обновляем прогресс-бар
                    pbar.update(1)
                    pbar.set_postfix({
                        'Успешно': f"{len([r for r in results if r['success']])}",
                        'С ограничениями': f"{len([r for r in results if r['constraints_passed']])}"
                    })
                    
                except Exception as e:
                    print(f"Ошибка при выполнении задачи: {e}")
                    pbar.update(1)
    
    # Создаем DataFrame
    df_results = pd.DataFrame(results)
    print(f"\n" + "="*60)
    print(f"РАСЧЕТЫ ЗАВЕРШЕНЫ")
    print(f"Всего расчетов: {len(df_results)}")
    print(f"Успешных расчетов: {df_results['success'].sum()}")
    print(f"Расчетов, прошедших все ограничения: {df_results['constraints_passed'].sum()}")
    
    # Выводим только успешные расчеты
    if df_results['success'].sum() > 0:
        successful_results = df_results[df_results['success']]
        print(f"\nУСПЕШНЫЕ РАСЧЕТЫ ({len(successful_results)}):")
        print(successful_results[['powder', 'k', 'delta', 'omega', 'W_0', 'velocity', 'max_pressure', 'barrel_length_calibers', 'constraints_passed']].head(10))
    else:
        print("\nНет успешных расчетов")
    
    # Выводим лучшие результаты по скорости (только прошедшие ограничения)
    if df_results['constraints_passed'].sum() > 0:
        best_results = df_results[(df_results['success']) & (df_results['constraints_passed'])].nlargest(10, 'velocity')
        print(f"\nТОП-10 РЕЗУЛЬТАТОВ ПО СКОРОСТИ (прошедшие все ограничения):")
        print(best_results[['powder', 'k', 'delta', 'omega', 'W_0', 'velocity', 'max_pressure', 'barrel_length_calibers']])
    else:
        print("\nНет результатов, прошедших все ограничения")
    
    # Анализ нарушений ограничений
    if len(df_results) > 0:
        print(f"\nАНАЛИЗ НАРУШЕНИЙ ОГРАНИЧЕНИЙ:")
        successful_calc = df_results[df_results['success']]
        if len(successful_calc) > 0:
            constraint_violations = successful_calc['constraints_violated'].value_counts()
            print(constraint_violations)
            
            print(f"\nСТАТИСТИКА ОГРАНИЧЕНИЙ:")
            print(f"Превышение давления (>390 МПа): {(successful_calc['max_pressure'] > MAX_PRESSURE).sum()}")
            print(f"Превышение длины ствола (>65 калибров): {(successful_calc['barrel_length'] > MAX_BARREL_LENGTH).sum()}")
            print(f"Все ограничения пройдены: {successful_calc['constraints_passed'].sum()}")
        else:
            print("Нет успешных расчетов для анализа")

if __name__ == "__main__":
    main()