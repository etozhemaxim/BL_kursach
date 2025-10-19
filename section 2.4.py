from pyballistics import ozvb_termo, get_options_sample
import numpy as np
from tqdm import tqdm
import itertools
import gc
import psutil
import os

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

def generate_mixture_name(powder_type, powders_info, omega_sum):
    if powder_type == "Свободная":
        name_parts = ["С"]
        for powder_idx, fraction in powders_info:
            name_parts.append(f"{powder_idx}/{fraction}")
        name_parts.append(str(round(omega_sum, 1)))
        return " ".join(name_parts)
    else:
        type_code = ""
        if powder_type == "ПО": type_code = "П1"
        elif powder_type == "ПС": type_code = "П7" 
        elif powder_type == "БО": type_code = "Б1"
        
        name_parts = [type_code]
        for powder_idx, fraction in powders_info:
            name_parts.append(f"{powder_idx}/{fraction}")
        name_parts.append(str(round(omega_sum, 1)))
        return " ".join(name_parts)

def calculate_mixture(opts, target_velocity=950):
    """Расчет смеси с оптимизацией памяти"""
    try:
        result = ozvb_termo(opts)
        
        if result and isinstance(result, dict):
            v_p_array = result.get('v_p', [])
            p_m_array = result.get('p_m', [])
            x_p_array = result.get('x_p', [])
            
            if len(v_p_array) == 0 or len(p_m_array) == 0 or len(x_p_array) == 0:
                return None
            
            # Находим индекс, когда скорость достигла целевой
            v_p_final = v_p_array[-1]
            if v_p_final < target_velocity:
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
            
            return {
                'p_max': p_max,
                'v_pm': v_pm,
                'x_pm': x_pm
            }
            
    except Exception as e:
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

def calculate_2powder_mixture(baza, powder_type, omega_sum_range, delta_range, alpha_range):
    """Расчет для смесей из 2 порохов с оптимизацией памяти"""
    results = []
    
    if powder_type == "Свободная":
        powders_list = baza.get_all_powders()
    else:
        powders_list = baza.get_powders_by_type(powder_type)
    
    if len(powders_list) < 2:
        return results
    
    powder_combinations = list(itertools.combinations(powders_list, 2))
    total_iterations = len(omega_sum_range) * len(delta_range) * len(alpha_range) * len(powder_combinations)
    
    # Ограничиваем количество комбинаций для тестирования
    max_combinations = 30
    if len(powder_combinations) > max_combinations:
        powder_combinations = powder_combinations[:max_combinations]
    
    iteration_count = 0
    gc_collect_interval = 100
    
    with tqdm(total=total_iterations, desc=f"2 пороха {powder_type}") as pbar:
        for omega_sum in omega_sum_range:
            q = 5.0
            if omega_sum / q < 0.01 or omega_sum / q > 1.0:
                pbar.update(len(delta_range) * len(alpha_range) * len(powder_combinations))
                continue
                
            for delta in delta_range:
                W_0 = omega_sum / delta
                
                for alpha in alpha_range:
                    if alpha < 0.1 or alpha > 0.9:
                        pbar.update(len(powder_combinations))
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
                        opts['stop_conditions']['v_p'] = 1000
                        opts['stop_conditions']['x_p'] = 10
                        
                        ballistics_result = calculate_mixture(opts)
                        
                        if ballistics_result:
                            p_max = ballistics_result['p_max']
                            v_pm = ballistics_result['v_pm']
                            x_pm = ballistics_result['x_pm']
                            
                            if (p_max <= 390000000.0 and 
                                v_pm >= 950 and 
                                x_pm >= 1.5):
                                
                                if powder_type == "Свободная":
                                    powder1_idx = baza.get_powder_index(powder1)
                                    powder2_idx = baza.get_powder_index(powder2)
                                    powders_info = [
                                        (powder1_idx, int(1/alpha)),
                                        (powder2_idx, int(1/(1-alpha)))
                                    ]
                                else:
                                    powder1_idx = powders_list.index(powder1) + 1
                                    powder2_idx = powders_list.index(powder2) + 1
                                    powders_info = [
                                        (powder1_idx, int(1/alpha)),
                                        (powder2_idx, int(1/(1-alpha)))
                                    ]
                                
                                mixture_name = generate_mixture_name(powder_type, powders_info, omega_sum)
                                
                                result_data = {
                                    'name': mixture_name,
                                    'p_max': p_max,
                                    'v_pm': v_pm,
                                    'x_pm': x_pm,
                                    'omega_sum': omega_sum,
                                    'W_0': W_0,
                                    'delta': delta,
                                    'powders': [powder1, powder2],
                                    'fractions': [alpha, 1-alpha],
                                    'type': powder_type,
                                    'num_powders': 2
                                }
                                
                                results.append(result_data)
                                save_result_to_file(result_data)
                        
                        iteration_count += 1
                        if iteration_count % gc_collect_interval == 0:
                            gc.collect()
                            memory_usage = check_memory_usage()
                            if memory_usage > 500:
                                pbar.set_postfix(memory=f"{memory_usage:.1f}MB")
                        
                        pbar.update(1)
    
    return results

def calculate_3powder_mixture(baza, powder_type, omega_sum_range, delta_range, fractions_range):
    """Расчет для смесей из 3 порохов с оптимизацией памяти"""
    results = []
    
    if powder_type == "Свободная":
        powders_list = baza.get_all_powders()
    else:
        powders_list = baza.get_powders_by_type(powder_type)
    
    if len(powders_list) < 3:
        return results
    
    powder_combinations = list(itertools.combinations(powders_list, 3))
    
    max_combinations = 15
    if len(powder_combinations) > max_combinations:
        powder_combinations = powder_combinations[:max_combinations]
    
    valid_fractions = []
    for f1 in fractions_range:
        for f2 in fractions_range:
            f3 = 1 - f1 - f2
            if f3 > 0.1 and f3 < 0.9:
                valid_fractions.append((f1, f2, f3))
    
    total_iterations = len(omega_sum_range) * len(delta_range) * len(valid_fractions) * len(powder_combinations)
    
    iteration_count = 0
    gc_collect_interval = 50
    
    with tqdm(total=total_iterations, desc=f"3 пороха {powder_type}") as pbar:
        for omega_sum in omega_sum_range:
            q = 5.0
            if omega_sum / q < 0.01 or omega_sum / q > 1.0:
                pbar.update(len(delta_range) * len(valid_fractions) * len(powder_combinations))
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
                        opts['stop_conditions']['v_p'] = 1000
                        opts['stop_conditions']['x_p'] = 10
                        
                        ballistics_result = calculate_mixture(opts)
                        
                        if ballistics_result:
                            p_max = ballistics_result['p_max']
                            v_pm = ballistics_result['v_pm']
                            x_pm = ballistics_result['x_pm']
                            
                            if (p_max <= 390000000.0 and 
                                v_pm >= 950 and 
                                x_pm >= 1.5):
                                
                                if powder_type == "Свободная":
                                    powder1_idx = baza.get_powder_index(powder1)
                                    powder2_idx = baza.get_powder_index(powder2)
                                    powder3_idx = baza.get_powder_index(powder3)
                                    powders_info = [
                                        (powder1_idx, int(1/f1)),
                                        (powder2_idx, int(1/f2)),
                                        (powder3_idx, int(1/f3))
                                    ]
                                else:
                                    powder1_idx = powders_list.index(powder1) + 1
                                    powder2_idx = powders_list.index(powder2) + 1
                                    powder3_idx = powders_list.index(powder3) + 1
                                    powders_info = [
                                        (powder1_idx, int(1/f1)),
                                        (powder2_idx, int(1/f2)),
                                        (powder3_idx, int(1/f3))
                                    ]
                                
                                mixture_name = generate_mixture_name(powder_type, powders_info, omega_sum)
                                
                                result_data = {
                                    'name': mixture_name,
                                    'p_max': p_max,
                                    'v_pm': v_pm,
                                    'x_pm': x_pm,
                                    'omega_sum': omega_sum,
                                    'W_0': W_0,
                                    'delta': delta,
                                    'powders': [powder1, powder2, powder3],
                                    'fractions': [f1, f2, f3],
                                    'type': powder_type,
                                    'num_powders': 3
                                }
                                
                                results.append(result_data)
                                save_result_to_file(result_data)
                        
                        iteration_count += 1
                        if iteration_count % gc_collect_interval == 0:
                            gc.collect()
                        
                        pbar.update(1)
    
    return results

def calculate_4powder_mixture(baza, powder_type, omega_sum_range, delta_range, fractions_range):
    """Расчет для смесей из 4 порохов с оптимизацией памяти"""
    results = []
    
    if powder_type == "Свободная":
        powders_list = baza.get_all_powders()
    else:
        powders_list = baza.get_powders_by_type(powder_type)
    
    if len(powders_list) < 4:
        return results
    
    # Сильно ограничиваем комбинации для 4 порохов
    powder_combinations = list(itertools.combinations(powders_list, 4))
    max_combinations = 8  # Очень мало комбинаций из-за сложности
    if len(powder_combinations) > max_combinations:
        powder_combinations = powder_combinations[:max_combinations]
    
    # Генерируем ограниченное количество валидных комбинаций долей
    valid_fractions = []
    fraction_values = fractions_range
    
    # Используем только симметричные распределения для уменьшения комбинаций
    symmetric_fractions = [
        (0.25, 0.25, 0.25, 0.25),  # равные доли
        (0.4, 0.2, 0.2, 0.2),      # один доминирующий
        (0.3, 0.3, 0.2, 0.2),      # два основных
        (0.35, 0.25, 0.2, 0.2),    # градиент
        (0.4, 0.3, 0.2, 0.1),      # убывающие доли
    ]
    
    # Добавляем несколько случайных комбинаций
    for _ in range(5):
        fractions = np.random.dirichlet(np.ones(4), size=1)[0]
        # Округляем до 0.05 для читаемости
        fractions = np.round(fractions / 0.05) * 0.05
        if np.sum(fractions) == 1.0 and all(f > 0.05 for f in fractions):
            valid_fractions.append(tuple(fractions))
    
    valid_fractions.extend(symmetric_fractions)
    
    total_iterations = len(omega_sum_range) * len(delta_range) * len(valid_fractions) * len(powder_combinations)
    
    iteration_count = 0
    gc_collect_interval = 20  # Чаще собираем мусор для 4 порохов
    
    with tqdm(total=total_iterations, desc=f"4 пороха {powder_type}") as pbar:
        for omega_sum in omega_sum_range:
            q = 5.0
            if omega_sum / q < 0.01 or omega_sum / q > 1.0:
                pbar.update(len(delta_range) * len(valid_fractions) * len(powder_combinations))
                continue
                
            for delta in delta_range:
                W_0 = omega_sum / delta
                
                for f1, f2, f3, f4 in valid_fractions:
                    # Проверяем, что сумма долей равна 1 с небольшой погрешностью
                    if abs(f1 + f2 + f3 + f4 - 1.0) > 0.01:
                        pbar.update(len(powder_combinations))
                        continue
                        
                    for powder1, powder2, powder3, powder4 in powder_combinations:
                        omega1 = omega_sum * f1
                        omega2 = omega_sum * f2
                        omega3 = omega_sum * f3
                        omega4 = omega_sum * f4
                        
                        # Проверяем минимальные массы порохов
                        if any(omega < 0.001 for omega in [omega1, omega2, omega3, omega4]):
                            pbar.update(1)
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
                        opts['stop_conditions']['v_p'] = 1000
                        opts['stop_conditions']['x_p'] = 10
                        
                        ballistics_result = calculate_mixture(opts)
                        
                        if ballistics_result:
                            p_max = ballistics_result['p_max']
                            v_pm = ballistics_result['v_pm']
                            x_pm = ballistics_result['x_pm']
                            
                            if (p_max <= 390000000.0 and 
                                v_pm >= 950 and 
                                x_pm >= 1.5):
                                
                                if powder_type == "Свободная":
                                    powder1_idx = baza.get_powder_index(powder1)
                                    powder2_idx = baza.get_powder_index(powder2)
                                    powder3_idx = baza.get_powder_index(powder3)
                                    powder4_idx = baza.get_powder_index(powder4)
                                    powders_info = [
                                        (powder1_idx, int(1/f1)),
                                        (powder2_idx, int(1/f2)),
                                        (powder3_idx, int(1/f3)),
                                        (powder4_idx, int(1/f4))
                                    ]
                                else:
                                    powder1_idx = powders_list.index(powder1) + 1
                                    powder2_idx = powders_list.index(powder2) + 1
                                    powder3_idx = powders_list.index(powder3) + 1
                                    powder4_idx = powders_list.index(powder4) + 1
                                    powders_info = [
                                        (powder1_idx, int(1/f1)),
                                        (powder2_idx, int(1/f2)),
                                        (powder3_idx, int(1/f3)),
                                        (powder4_idx, int(1/f4))
                                    ]
                                
                                mixture_name = generate_mixture_name(powder_type, powders_info, omega_sum)
                                
                                result_data = {
                                    'name': mixture_name,
                                    'p_max': p_max,
                                    'v_pm': v_pm,
                                    'x_pm': x_pm,
                                    'omega_sum': omega_sum,
                                    'W_0': W_0,
                                    'delta': delta,
                                    'powders': [powder1, powder2, powder3, powder4],
                                    'fractions': [f1, f2, f3, f4],
                                    'type': powder_type,
                                    'num_powders': 4
                                }
                                
                                results.append(result_data)
                                save_result_to_file(result_data)
                                print(f"  Найдена 4-компонентная смесь: {mixture_name}")
                        
                        iteration_count += 1
                        if iteration_count % gc_collect_interval == 0:
                            gc.collect()
                            memory_usage = check_memory_usage()
                            pbar.set_postfix(memory=f"{memory_usage:.1f}MB")
                        
                        pbar.update(1)
    
    return results

def initialize_results_file():
    """Инициализация файла результатов с заголовками"""
    with open("results.txt", "w", encoding="utf-8") as f:
        f.write("name,p_max_mpa,v_pm_mps,x_pm_m,omega_sum_kg,W_0_m3,delta_kg_m3,type,num_powders\n")

def main():
    """Основная функция расчета с сохранением в файл"""
    baza = BazaPowder()
    
    # Инициализируем файл результатов
    initialize_results_file()
    
    # Уменьшаем диапазоны для оптимизации
    omega_sum_range = np.linspace(0.1, 3.0, 4)  # Еще меньше точек
    delta_range = np.linspace(900, 1400, 4)     # Уменьшенный диапазон
    alpha_range = np.linspace(0.3, 0.7, 3)      # для 2 порохов
    fractions_range_3 = np.linspace(0.2, 0.6, 3)  # для 3 порохов
    fractions_range_4 = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]  # для 4 порохов
    
    all_results = []
    
    mixture_types = ["ПО", "ПС", "БО", "Свободная"]
    
    print("=== НАЧАЛО РАСЧЕТА С 4-КОМПОНЕНТНЫМИ СМЕСЯМИ ===")
    print(f"Использование памяти в начале: {check_memory_usage():.1f} MB")
    print("Результаты сохраняются в файл: results.txt")
    
    for powder_type in mixture_types:
        print(f"\n=== Расчет смесей типа: {powder_type} ===")
        
        # Расчет для 2 порохов
        print("Запуск расчета для 2 порохов...")
        results_2p = calculate_2powder_mixture(baza, powder_type, omega_sum_range, delta_range, alpha_range)
        all_results.extend(results_2p)
        print(f"Найдено решений для 2 порохов: {len(results_2p)}")
        
        gc.collect()
        print(f"Память после 2 порохов: {check_memory_usage():.1f} MB")
        
        # Расчет для 3 порохов
        print("Запуск расчета для 3 порохов...")
        results_3p = calculate_3powder_mixture(baza, powder_type, omega_sum_range, delta_range, fractions_range_3)
        all_results.extend(results_3p)
        print(f"Найдено решений для 3 порохов: {len(results_3p)}")
        
        gc.collect()
        print(f"Память после 3 порохов: {check_memory_usage():.1f} MB")
        
        # Расчет для 4 порохов
        print("Запуск расчета для 4 порохов...")
        results_4p = calculate_4powder_mixture(baza, powder_type, omega_sum_range, delta_range, fractions_range_4)
        all_results.extend(results_4p)
        print(f"Найдено решений для 4 порохов: {len(results_4p)}")
        
        gc.collect()
        print(f"Память после 4 порохов: {check_memory_usage():.1f} MB")
    
    # Вывод итогов
    print(f"\n=== ИТОГОВЫЕ РЕЗУЛЬТАТЫ ===")
    print(f"Всего найдено валидных решений: {len(all_results)}")
    
    # Статистика по количеству порохов
    counts = {2: 0, 3: 0, 4: 0}
    for result in all_results:
        counts[result['num_powders']] += 1
    
    print(f"Распределение по количеству порохов:")
    for num_powders, count in counts.items():
        print(f"  {num_powders} пороха: {count} решений")
    
    print(f"Использование памяти в конце: {check_memory_usage():.1f} MB")
    print("Все результаты сохранены в файл: results.txt")
    
    if all_results:
        # Вывод топ-3 результатов по скорости для каждого количества порохов
        print(f"\nЛучшие результаты по скорости:")
        for num_powders in [2, 3, 4]:
            powder_results = [r for r in all_results if r['num_powders'] == num_powders]
            if powder_results:
                powder_results.sort(key=lambda x: x['v_pm'], reverse=True)
                print(f"\nТоп-3 для {num_powders} порохов:")
                for i, result in enumerate(powder_results[:3]):
                    print(f"  {i+1}. {result['name']}: "
                          f"P={result['p_max']/1e6:.1f}МПа, "
                          f"V={result['v_pm']:.0f}м/с, "
                          f"L={result['x_pm']:.2f}м")
    
    return all_results

if __name__ == "__main__":
    gc.enable()
    gc.set_threshold(500, 10, 10)  # Более агрессивная сборка мусора
    
    results = main()