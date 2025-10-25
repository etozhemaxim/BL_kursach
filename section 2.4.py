from pyballistics import ozvb_termo, get_options_sample
import numpy as np
from tqdm import tqdm
import itertools
import gc
import psutil
import os
import time
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

class BazaPowder:
    def __init__(self):
        self.powders = {
            'ПО': ['4/1 фл', '5/1', '7/1 УГ', '37/1 тр', '8/1 УГ'],
            'ПС': ['4/7', '5/7 н/а', '14/7', '17/7', '22/7'],
            'БО': ['ДРП', 'ВТ', 'ДГ-4 13/1', 'ДГ-4 15/1', 'НДТ-3 19/1']
        }
        
        self.all_powders = []
        for category in self.powders.values():
            self.all_powders.extend(category)
    
    def get_powders_by_type(self, powder_type):
        return self.powders.get(powder_type, [])
    
    def get_all_powders(self):
        return self.all_powders
    
    def get_powder_index(self, powder_name):
        try:
            return self.all_powders.index(powder_name) + 1
        except ValueError:
            return -1

# Глобальные переменные для статистики
calculation_stats = []

def generate_mixture_name(powder_type, powders_info, omega_sum):
    """
    Генерация имени смеси с фактическими долями (без округления)
    powders_info: список кортежей (индекс_пороха, фактическая_доля)
    """
    if powder_type == "Свободная":
        name_parts = ["С"]
        for powder_idx, actual_fraction in powders_info:
            # Сохраняем фактическую долю с точностью до 3 знаков
            name_parts.append(f"{powder_idx}/{actual_fraction:.3f}")
        name_parts.append(str(round(omega_sum, 1)))
        return " ".join(name_parts)
    else:
        type_code = ""
        if powder_type == "ПО": type_code = "П1"
        elif powder_type == "ПС": type_code = "П7" 
        elif powder_type == "БО": type_code = "Б1"
        
        name_parts = [type_code]
        for powder_idx, actual_fraction in powders_info:
            # Сохраняем фактическую долю с точностью до 3 знаков
            name_parts.append(f"{powder_idx}/{actual_fraction:.3f}")
        name_parts.append(str(round(omega_sum, 1)))
        return " ".join(name_parts)

def calculate_single_mixture(params):
    """Расчет одной смеси в изолированном процессе"""
    opts, target_velocity = params
    
    try:
        result = ozvb_termo(opts)
        
        if result and isinstance(result, dict):
            v_p_array = result.get('v_p', [])
            p_m_array = result.get('p_m', [])
            x_p_array = result.get('x_p', [])
            
            if len(v_p_array) == 0 or len(p_m_array) == 0 or len(x_p_array) == 0:
                return None
            
            # Находим момент достижения целевой скорости
            target_idx = None
            for i, v in enumerate(v_p_array):
                if v >= target_velocity:
                    target_idx = i
                    break
            
            if target_idx is None:
                return None
            
            p_max = np.max(p_m_array[:target_idx+1])
            x_pm = x_p_array[target_idx]
            v_pm = v_p_array[target_idx]
            
            if (p_max <= 390000000.0 and v_pm >= 950 and x_pm >= 1.5):
                return {
                    'p_max': float(p_max),
                    'v_pm': float(v_pm),
                    'x_pm': float(x_pm)
                }
                
    except Exception:
        return None
    
    return None

def check_memory_usage():
    """Проверка использования памяти"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # в МБ

def save_result_to_file(result, filename="results.txt"):
    """Сохранение результата в файл"""
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"{result['name']},"
                f"{result['p_max']/1e6:.2f},"
                f"{result['v_pm']:.2f},"
                f"{result['x_pm']:.3f},"
                f"{result['omega_sum']:.3f},"
                f"{result['W_0']:.6f},"
                f"{result['delta']:.1f},"
                f"{result['type']},"
                f"{result['num_powders']}\n")

def save_time_stats_to_file(stats, filename="time.txt"):
    """Сохранение статистики по времени в файл"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write("=== СТАТИСТИКА ВРЕМЕНИ РАСЧЕТА ===\n")
        f.write(f"Дата расчета: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n")
        f.write("Тип_пороха,Кол-во_порохов,Комбинации_порохов,Итерации,Время_сек,Среднее_время_итерации_сек\n")
        
        total_time = 0
        total_iterations = 0
        
        for stat in stats:
            powder_type = stat['powder_type']
            num_powders = stat['num_powders']
            powder_combinations = stat['powder_combinations']
            iterations = stat['iterations']
            calculation_time = stat['calculation_time']
            avg_time_per_iteration = stat['avg_time_per_iteration']
            
            f.write(f"{powder_type},{num_powders},{powder_combinations},{iterations},"
                   f"{calculation_time:.2f},{avg_time_per_iteration:.4f}\n")
            
            total_time += calculation_time
            total_iterations += iterations
        
        f.write("=" * 50 + "\n")
        f.write(f"ОБЩИЕ ПОКАЗАТЕЛИ:\n")
        f.write(f"Всего итераций: {total_iterations}\n")
        f.write(f"Общее время расчета: {total_time:.2f} сек\n")
        f.write(f"Среднее время на итерацию: {total_time/total_iterations:.4f} сек\n")
        f.write(f"Всего комбинаций рассчитано: {len(stats)}\n")

def run_calculations_in_processes(tasks, task_info, baza, powder_type, desc):
    """Запуск расчетов в процессах с прогресс-баром"""
    results = []
    total_iterations = len(tasks)
    successful_calculations = 0
    start_time = time.time()
    initial_memory = check_memory_usage()
    
    print(f"  {desc}: {total_iterations} расчетов в процессах...")
    
    # Запускаем расчеты в процессах
    with ProcessPoolExecutor(max_workers=4) as executor:
        future_to_task = {executor.submit(calculate_single_mixture, task): i 
                         for i, task in enumerate(tasks)}
        
        with tqdm(total=total_iterations, desc=desc) as pbar:
            for future in as_completed(future_to_task):
                i = future_to_task[future]
                try:
                    ballistics_result = future.result()
                    if ballistics_result:
                        info = task_info[i]
                        
                        # Генерация имени смеси с фактическими долями
                        if info['powder_type'] == "Свободная":
                            powders_info = []
                            for powder, fraction in zip(info['powders'], info['fractions']):
                                powder_idx = baza.get_powder_index(powder)
                                powders_info.append((powder_idx, fraction))  # Сохраняем фактическую долю
                        else:
                            powders_info = []
                            for powder, fraction in zip(info['powders'], info['fractions']):
                                powder_idx = info['powders_list'].index(powder) + 1
                                powders_info.append((powder_idx, fraction))  # Сохраняем фактическую долю
                        
                        mixture_name = generate_mixture_name(info['powder_type'], powders_info, info['omega_sum'])
                        
                        result_data = {
                            'name': mixture_name,
                            'p_max': ballistics_result['p_max'],
                            'v_pm': ballistics_result['v_pm'],
                            'x_pm': ballistics_result['x_pm'],
                            'omega_sum': info['omega_sum'],
                            'W_0': info['W_0'],
                            'delta': info['delta'],
                            'powders': info['powders'],
                            'fractions': info['fractions'],
                            'type': info['powder_type'],
                            'num_powders': info['num_powders']
                        }
                        
                        results.append(result_data)
                        save_result_to_file(result_data)
                        successful_calculations += 1
                        
                except Exception as e:
                    pass
                
                pbar.update(1)
                
                # Периодическая очистка памяти
                if pbar.n % 100 == 0:
                    gc.collect()
    
    end_time = time.time()
    calculation_time = end_time - start_time
    
    # Сохраняем статистику
    stats = {
        'powder_type': powder_type,
        'num_powders': task_info[0]['num_powders'] if task_info else 0,
        'powder_combinations': len(set(tuple(info['powders']) for info in task_info)) if task_info else 0,
        'iterations': total_iterations,
        'calculation_time': calculation_time,
        'avg_time_per_iteration': calculation_time / total_iterations if total_iterations > 0 else 0,
        'successful_calculations': successful_calculations
    }
    calculation_stats.append(stats)
    
    final_memory = check_memory_usage()
    print(f"  Статистика {desc}: {total_iterations} итераций, "
          f"{calculation_time:.2f} сек, {successful_calculations} успешных, "
          f"память: {initial_memory:.1f}MB -> {final_memory:.1f}MB")
    
    return results

def calculate_2powder_mixture(baza, powder_type, omega_sum_range, delta_range, alpha_range):
    """Расчет для смесей из 2 порохов"""
    results = []
    
    if powder_type == "Свободная":
        powders_list = baza.get_all_powders()
    else:
        powders_list = baza.get_powders_by_type(powder_type)
    
    if len(powders_list) < 2:
        return results
    
    powder_combinations = list(itertools.combinations(powders_list, 2))
    
    # Подготавливаем все задачи для расчета
    tasks = []
    task_info = []
    
    for omega_sum in omega_sum_range:
        q = 5.0
        if omega_sum / q < 0.01 or omega_sum / q > 1.0:
            continue
            
        for delta in delta_range:
            W_0 = omega_sum / delta
            
            for alpha in alpha_range:
                if alpha < 0.1 or alpha > 0.9:
                    continue
                    
                for powder1, powder2 in powder_combinations:
                    omega1 = omega_sum * alpha
                    omega2 = omega_sum * (1 - alpha)
                    
                    opts = get_options_sample()
                    opts['powders'] = [
                        {'omega': omega1, 'dbname': powder1},
                        {'omega': omega2, 'dbname': powder2}
                    ]
                    opts['init_conditions']['q'] = q
                    opts['init_conditions']['d'] = 0.085
                    opts['init_conditions']['W_0'] = W_0
                    opts['init_conditions']['phi_1'] = 1.04
                    opts['init_conditions']['p_0'] = 30000000.0
                    opts['stop_conditions']['v_p'] = 950
                    opts['stop_conditions']['x_p'] = 5
                    
                    tasks.append((opts, 950))
                    task_info.append({
                        'powder_type': powder_type,
                        'powders': [powder1, powder2],
                        'fractions': [alpha, 1-alpha],  # Фактические доли
                        'omega_sum': omega_sum,
                        'W_0': W_0,
                        'delta': delta,
                        'num_powders': 2,
                        'powders_list': powders_list if powder_type != "Свободная" else baza.get_all_powders()
                    })
    
    return run_calculations_in_processes(tasks, task_info, baza, powder_type, f"2 пороха {powder_type}")

def calculate_3powder_mixture(baza, powder_type, omega_sum_range, delta_range, fractions_range):
    """Расчет для смесей из 3 порохов"""
    results = []
    
    if powder_type == "Свободная":
        powders_list = baza.get_all_powders()
    else:
        powders_list = baza.get_powders_by_type(powder_type)
    
    if len(powders_list) < 3:
        return results
    
    powder_combinations = list(itertools.combinations(powders_list, 3))
    
    # Генерируем валидные комбинации долей
    valid_fractions = []
    for f1 in fractions_range:
        for f2 in fractions_range:
            f3 = 1 - f1 - f2
            if f3 > 0.1 and f3 < 0.9:
                valid_fractions.append((f1, f2, f3))
    
    # Подготавливаем все задачи для расчета
    tasks = []
    task_info = []
    
    for omega_sum in omega_sum_range:
        q = 5.0
        if omega_sum / q < 0.01 or omega_sum / q > 1.0:
            continue
            
        for delta in delta_range:
            W_0 = omega_sum / delta
            
            for f1, f2, f3 in valid_fractions:
                for powder1, powder2, powder3 in powder_combinations:
                    omega1 = omega_sum * f1
                    omega2 = omega_sum * f2
                    omega3 = omega_sum * f3
                    
                    opts = get_options_sample()
                    opts['powders'] = [
                        {'omega': omega1, 'dbname': powder1},
                        {'omega': omega2, 'dbname': powder2},
                        {'omega': omega3, 'dbname': powder3}
                    ]
                    opts['init_conditions']['q'] = q
                    opts['init_conditions']['d'] = 0.085
                    opts['init_conditions']['W_0'] = W_0
                    opts['init_conditions']['phi_1'] = 1.04
                    opts['init_conditions']['p_0'] = 30000000.0
                    opts['stop_conditions']['v_p'] = 950
                    opts['stop_conditions']['x_p'] = 5
                    
                    tasks.append((opts, 950))
                    task_info.append({
                        'powder_type': powder_type,
                        'powders': [powder1, powder2, powder3],
                        'fractions': [f1, f2, f3],  # Фактические доли
                        'omega_sum': omega_sum,
                        'W_0': W_0,
                        'delta': delta,
                        'num_powders': 3,
                        'powders_list': powders_list if powder_type != "Свободная" else baza.get_all_powders()
                    })
    
    return run_calculations_in_processes(tasks, task_info, baza, powder_type, f"3 пороха {powder_type}")

def calculate_4powder_mixture(baza, powder_type, omega_sum_range, delta_range, fractions_range):
    """Расчет для смесей из 4 порохов"""
    results = []
    
    if powder_type == "Свободная":
        powders_list = baza.get_all_powders()
    else:
        powders_list = baza.get_powders_by_type(powder_type)
    
    if len(powders_list) < 4:
        return results
    
    powder_combinations = list(itertools.combinations(powders_list, 4))
    
    # Генерируем ограниченное количество валидных комбинаций долей
    valid_fractions = []
    
    # Симметричные распределения
    symmetric_fractions = [
        (0.25, 0.25, 0.25, 0.25),
        (0.4, 0.2, 0.2, 0.2),
        (0.3, 0.3, 0.2, 0.2),
        (0.35, 0.25, 0.2, 0.2),
        (0.4, 0.3, 0.2, 0.1),
    ]
    
    # Добавляем несколько случайных комбинаций
    for _ in range(5):
        fractions = np.random.dirichlet(np.ones(4), size=1)[0]
        fractions = np.round(fractions / 0.05) * 0.05
        if np.sum(fractions) == 1.0 and all(f > 0.05 for f in fractions):
            valid_fractions.append(tuple(fractions))
    
    valid_fractions.extend(symmetric_fractions)
    
    # Подготавливаем все задачи для расчета
    tasks = []
    task_info = []
    
    for omega_sum in omega_sum_range:
        q = 5.0
        if omega_sum / q < 0.01 or omega_sum / q > 1.0:
            continue
            
        for delta in delta_range:
            W_0 = omega_sum / delta
            
            for f1, f2, f3, f4 in valid_fractions:
                if abs(f1 + f2 + f3 + f4 - 1.0) > 0.01:
                    continue
                    
                for powder1, powder2, powder3, powder4 in powder_combinations:
                    omega1 = omega_sum * f1
                    omega2 = omega_sum * f2
                    omega3 = omega_sum * f3
                    omega4 = omega_sum * f4
                    
                    if any(omega < 0.001 for omega in [omega1, omega2, omega3, omega4]):
                        continue
                    
                    opts = get_options_sample()
                    opts['powders'] = [
                        {'omega': omega1, 'dbname': powder1},
                        {'omega': omega2, 'dbname': powder2},
                        {'omega': omega3, 'dbname': powder3},
                        {'omega': omega4, 'dbname': powder4}
                    ]
                    opts['init_conditions']['q'] = q
                    opts['init_conditions']['d'] = 0.085
                    opts['init_conditions']['W_0'] = W_0
                    opts['init_conditions']['phi_1'] = 1.04
                    opts['init_conditions']['p_0'] = 30000000.0
                    opts['stop_conditions']['v_p'] = 950
                    opts['stop_conditions']['x_p'] = 5
                    
                    tasks.append((opts, 950))
                    task_info.append({
                        'powder_type': powder_type,
                        'powders': [powder1, powder2, powder3, powder4],
                        'fractions': [f1, f2, f3, f4],  # Фактические доли
                        'omega_sum': omega_sum,
                        'W_0': W_0,
                        'delta': delta,
                        'num_powders': 4,
                        'powders_list': powders_list if powder_type != "Свободная" else baza.get_all_powders()
                    })
    
    return run_calculations_in_processes(tasks, task_info, baza, powder_type, f"4 пороха {powder_type}")

def initialize_results_file():
    """Инициализация файла результатов с заголовками"""
    with open("results.txt", "w", encoding="utf-8") as f:
        f.write("name,p_max_mpa,v_pm_mps,x_pm_m,omega_sum_kg,W_0_m3,delta_kg_m3,type,num_powders\n")

def main():
    """Основная функция расчета"""
    global calculation_stats
    calculation_stats = []
    
    baza = BazaPowder()
    initialize_results_file()
    
    # Диапазоны параметров
    omega_sum_range = np.linspace(0.1, 3.0, 4)
    delta_range = np.linspace(900, 1400, 5)
    alpha_range = np.linspace(0.3, 0.7, 3)
    fractions_range_3 = np.linspace(0.2, 0.6, 3)
    fractions_range_4 = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    
    all_results = []
    
    mixture_types = ["ПО", "ПС", "БО", "Свободная"]
    
    total_start_time = time.time()
    

    
    for powder_type in mixture_types:
        print(f"\n=== Расчет смесей типа: {powder_type} ===")
        
        # Расчет для 2 порохов
        print("Запуск расчета для 2 порохов...")
        results_2p = calculate_2powder_mixture(baza, powder_type, omega_sum_range, delta_range, alpha_range)
        all_results.extend(results_2p)
        print(f"Найдено решений для 2 порохов: {len(results_2p)}")
        
        # Очистка памяти между этапами
        gc.collect()
        time.sleep(2)
        print(f"Память после 2 порохов: {check_memory_usage():.1f} MB")
        
        # Расчет для 3 порохов
        print("Запуск расчета для 3 порохов...")
        results_3p = calculate_3powder_mixture(baza, powder_type, omega_sum_range, delta_range, fractions_range_3)
        all_results.extend(results_3p)
        print(f"Найдено решений для 3 порохов: {len(results_3p)}")
        
        # Очистка памяти между этапами
        gc.collect()
        time.sleep(2)
        print(f"Память после 3 порохов: {check_memory_usage():.1f} MB")
        
        # Расчет для 4 порохов
        print("Запуск расчета для 4 порохов...")
        results_4p = calculate_4powder_mixture(baza, powder_type, omega_sum_range, delta_range, fractions_range_4)
        all_results.extend(results_4p)
        print(f"Найдено решений для 4 порохов: {len(results_4p)}")
        
        # Очистка памяти между типами порохов
        gc.collect()
        time.sleep(2)
        print(f"Память после {powder_type}: {check_memory_usage():.1f} MB")
    
    total_end_time = time.time()
    total_calculation_time = total_end_time - total_start_time
    
    save_time_stats_to_file(calculation_stats)
    
    # Вывод итогов
    print(f"\n=== ИТОГОВЫЕ РЕЗУЛЬТАТЫ ===")
    print(f"Всего найдено валидных решений: {len(all_results)}")
    print(f"Общее время расчета: {total_calculation_time:.2f} сек")
    
    # Статистика по количеству порохов
    counts = {2: 0, 3: 0, 4: 0}
    for result in all_results:
        counts[result['num_powders']] += 1
    
    print(f"Распределение по количеству порохов:")
    for num_powders, count in counts.items():
        print(f"  {num_powders} пороха: {count} решений")
    
    print(f"Использование памяти в конце: {check_memory_usage():.1f} MB")
    print("Все результаты сохранены в файл: results.txt")
    print("Статистика времени сохранена в файл: time.txt")
    
    if all_results:
        # Вывод топ-5 результатов по скорости для каждого количества порохов
        print(f"\nЛучшие результаты по скорости:")
        for num_powders in [2, 3, 4]:
            powder_results = [r for r in all_results if r['num_powders'] == num_powders]
            if powder_results:
                powder_results.sort(key=lambda x: x['v_pm'], reverse=True)
                print(f"\nТоп-5 для {num_powders} порохов:")
                for i, result in enumerate(powder_results[:5]):
                    print(f"  {i+1}. {result['name']}: "
                          f"P={result['p_max']/1e6:.1f}МПа, "
                          f"V={result['v_pm']:.0f}м/с, "
                          f"L={result['x_pm']:.2f}м")
    
    return all_results

if __name__ == "__main__":
    # Важно для multiprocessing в Windows
    mp.freeze_support()
    
    gc.enable()
    gc.collect()  # Очистка перед запуском
    
    try:
        results = main()
        print("\n🎉 РАСЧЕТ УСПЕШНО ЗАВЕРШЕН!")
    except Exception as e:
        print(f"!!! ОШИБКА: {e} !!!")
        import traceback
        traceback.print_exc()