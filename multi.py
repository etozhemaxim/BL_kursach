from pyballistics import ozvb_termo, get_options_sample, get_db_powder
import numpy as np
from tqdm import tqdm
import itertools
import copy
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
import time

# Параметры для варьирования
omega_sum_range = np.linspace(1, 5, 5)  # общая масса пороха в кг
delta_range = np.linspace(600,1200, 5)   # плотность заряжания
alpha_range = np.linspace(0.1, 0.9, 9)    # доля первого пороха (0.1-0.9)

class Powders:
    def __init__(self):
        self.powders = ['ДГ-4 14/1', 'ДГ-3 13/1', 'ДГ-4 15/1', 'АПЦ-235 П 16/1', 'ДГ-3 14/1', 'МАП-1 23/1', 'БНГ-1355 25/1', 'НДТ-3 16/1', 'ДГ-2 15/1', 'УГФ-1', 'УГ-1', 'ДГ-3 17/1', 'НДТ-2 16/1', 'НДТ-3 18/1', 'ДГ-3 18/1', 'ДГ-2 17/1', 'НДТ-3 19/1', 'ДГ-3 20/1', 'НДТ-2 19/1', '12/1 тр МН', '7/1 УГ', '15/1 тр В/А', '8/1 УГ', '16/1 тр В/А', '11/1 БП', '12/1 тр БП', '18/1 тр', '16/1 тр', '22/1 тр', '11/1 УГ', '12/1 УГ', '18/1 тр БП', '9/7 МН', '12/7', '14/7 В/А', '15/7', '9/7 БП', '14/7', '17/7', '14/7 БП']
    def get_powder_index(self, powder_name):
        """Получает индекс пороха в списке (начиная с 1)"""
        try:
            return self.powders.index(powder_name) + 1
        except ValueError:
            return -1

def get_powder_data(powder_name):
    """Получает данные пороха через get_db_powder"""
    try:
        return get_db_powder(powder_name)
    except Exception as e:
        print(f"Ошибка получения данных для пороха {powder_name}: {e}")
        return None

# Инициализация базы порохов
powders_DB = Powders()

available_powders = []
for p in powders_DB.powders[:]:
    powder_data = get_powder_data(p)
    if powder_data is not None:
        available_powders.append(p)

db = available_powders

# Базовые настройки для всех процессов
base_opts = get_options_sample()
base_opts['init_conditions']['q'] = 5
base_opts['init_conditions']['d'] = 0.085
base_opts['init_conditions']['phi_1'] = 1.04
base_opts['init_conditions']['p_0'] = 30000000.0
base_opts['stop_conditions']['v_p'] = 950
base_opts['stop_conditions']['p_max'] = 390000000.0
base_opts['stop_conditions']['x_p'] = 5.625

def generate_mixture_name(powder1, powder2, alpha, omega_sum, powders_db):
    """
    Генерация имени смеси в формате: С [индекс1]/[доля1] [индекс2]/[доля2] [масса]
    Пример: С 6/0.350 7/0.650 3.5
    """
    powder1_idx = powders_db.get_powder_index(powder1)
    powder2_idx = powders_db.get_powder_index(powder2)
    
    name_parts = ["С"]  # "С" означает смесь
    
    # Добавляем информацию о первом порохе и его доле
    name_parts.append(f"{powder1_idx}/{alpha:.1f}")
    
    # Добавляем информацию о втором порохе и его доле
    name_parts.append(f"{powder2_idx}/{(1-alpha):.1f}")
    
    # Добавляем общую массу (округляем до 1 знака)
    name_parts.append(str(round(omega_sum, 1)))
    
    return " ".join(name_parts)

def x_e_func(result):
    psi_1 = result['psi_1'] 
    psi_2 = result['psi_2'] 
    x_p = result['x_p']
    
    tolerance = 1e-3
    
    for i in range(len(psi_1)):
        if (abs(psi_1[i] - 1.0) < tolerance and abs(psi_2[i] - 1.0) < tolerance):
            return x_p[i]

    return x_p[-1]

def Z_b1_func(x_e, f_sum, omega_sum, vardelta, delta, b, k, result):
    phi_1 = 1.04
    p_ign = 5e6
    omega_ign = 0.01
    K = (1e-40)**2
    v_p = result['v_p'][-1]
    x_p = result['x_p'][-1]

    Pi = f_sum / (k - 1) 

    phi = phi_1 + (1 / (3 * base_opts['init_conditions']['q'])) * (omega_ign + omega_sum)   

    E_m = (base_opts['init_conditions']['q'] * v_p**2) / 2 

    eta_rm = phi * E_m / Pi    

    eta_re = E_m / x_p

    Lambda_e = x_e / x_p

    zeta = (p_ign / f_sum) * ((1 / vardelta) - (1 / delta)) * (1 / (1 + ((b * p_ign) / f_sum)))

    Lamda_m = (1 - b * vardelta * (1 + zeta) + Lambda_e) * ((1 + zeta + eta_re) / (1 + zeta - eta_rm))**(1 /(k-1)) - (1 - b * vardelta * (1 + zeta))

    N_s = K * (1 + Lamda_m) * (base_opts['init_conditions']['q'] / omega_sum)

    Z_b1 = (base_opts['init_conditions']['q'] * v_p**2 / 2)**5 * (np.sqrt(N_s) / omega_sum * x_p**4) 

    return Z_b1 , Pi, phi, E_m,  eta_rm,  eta_re, Lambda_e, zeta ,  Lamda_m , N_s , (1 - b * vardelta * (1 + zeta) + Lambda_e), ((1 + zeta + eta_re) / (1 + zeta - eta_rm))**(1 / (k - 1)), (1 - b * vardelta * (1 + zeta))

def save_solution_to_file(solution, filename="result2.txt"):
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"{solution['mixture_name']},{solution['powder1']},{solution['powder2']},"
                f"{solution['mass1']:.3f},{solution['mass2']:.3f},"
                f"{solution['alpha']:.3f},{solution['omega_sum']:.3f},"
                f"{solution['omega_q_ratio']:.3f},{solution['delta']:.1f},"
                f"{solution['W_0']:.6f},{solution['Z_b1']:.6e},"
                f"{solution['velocity']:.2f},{solution['velocity_cold']:.2f},{solution['velocity_hot']:.2f},"
                f"{solution['pressure_max']/1e6:.2f},{solution['pressure_hot']/1e6:.2f},"
                f"{solution['x_p']:.3f},{solution['x_e']:.3f},"
                f"{solution['f_sum']:.0f},{solution['delta_sum']:.1f},{solution['b_sum']:.6f},{solution['k_sum']:.3f}\n")

def initialize_result_file(filename="result2.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("mixture_name,powder1,powder2,mass1_kg,mass2_kg,alpha,omega_sum_kg,omega_q_ratio,delta_kg_m3,W0_m3,Z_b1,"
                "velocity_mps,velocity_cold_mps,velocity_hot_mps,pressure_max_mpa,pressure_hot_mpa,"
                "x_p_m,x_e_m,f_sum,delta_sum,b_sum,k_sum\n")

def run_simulation(params):
    """Запускает один расчет для комбинации параметров"""
    try:
        powder1, powder2, omega_sum, delta, alpha, base_opts_copy, powders_db = params
        
        # Проверяем базовые условия
        q = base_opts_copy['init_conditions']['q']
        if omega_sum / q < 0.1 or omega_sum / q > 0.8:
            return None
            
        W_0 = omega_sum / delta
        if W_0 <= 0:
            return None
            
        if alpha < 0.1 or alpha > 0.9:
            return None

        # Получаем данные порохов
        powder1_data = get_powder_data(powder1)
        powder2_data = get_powder_data(powder2)
        
        if powder1_data is None or powder2_data is None:
            return None

        mass1 = omega_sum * alpha
        mass2 = omega_sum * (1 - alpha)
        
        # Копируем настройки чтобы не менять оригинал
        opts = copy.deepcopy(base_opts_copy)
        opts['powders'] = [
            {'omega': mass1, 'dbname': powder1},
            {'omega': mass2, 'dbname': powder2}
        ]

        fraction1 = mass1 / omega_sum
        fraction2 = mass2 / omega_sum

        f_sum = alpha * powder1_data['f'] + (1 - alpha) * powder2_data['f']
        delta_sum = alpha * powder1_data['delta'] + (1 - alpha) * powder2_data['delta']
        b_sum = alpha * powder1_data['b'] + (1 - alpha) * powder2_data['b']
        k_sum = alpha * powder1_data['k'] + (1 - alpha) * powder2_data['k']

        opts['init_conditions']['W_0'] = W_0

        try:
            result_normal = ozvb_termo(opts)
            
            x_pm = result_normal['x_p'][-1]
            v_pm = result_normal['v_p'][-1]
            p_max = max(result_normal['p_m'])

            x_e = x_e_func(result_normal)
            Z_b1, Pi, phi, E_m, eta_rm, eta_re, Lambda_e, zeta, Lamda_m, N_s, a, b, c = Z_b1_func(x_e, f_sum, omega_sum, delta, delta_sum, b_sum, k_sum, result_normal)
            
            base_conditions_met = (p_max <= 390000000.0 and v_pm >= 950 and x_pm >= 1.5 and x_pm < 5.625)
            
            if base_conditions_met:
                # -50°C
                opts_cold = copy.deepcopy(opts)
                opts_cold['init_conditions']['T_0'] = 223.15
                opts_cold['stop_conditions']['v_p'] = 830
                result_cold = ozvb_termo(opts_cold)
                v_pm_cold = result_cold['v_p'][-1]
                
                if v_pm_cold >= 830:
                    # +50°C
                    opts_hot = copy.deepcopy(opts)
                    opts_hot['init_conditions']['T_0'] = 323.15
                    result_hot = ozvb_termo(opts_hot)
                    p_mz_hot = result_hot['p_m'][-1]
                    
                    if p_mz_hot <= 180000000:
                        # Генерируем имя смеси
                        mixture_name = generate_mixture_name(powder1, powder2, alpha, omega_sum, powders_db)
                        
                        solution = {
                            'mixture_name': mixture_name,
                            'powder1': powder1,
                            'powder2': powder2,
                            'mass1': mass1,
                            'mass2': mass2,
                            'alpha': alpha,
                            'omega_sum': omega_sum,
                            'omega_q_ratio': omega_sum / q,
                            'delta': delta,
                            'W_0': W_0,
                            'Z_b1': Z_b1,
                            'velocity': v_pm,
                            'velocity_cold': v_pm_cold,
                            'velocity_hot': result_hot['v_p'][-1],
                            'pressure_max': p_max,
                            'pressure_hot': p_mz_hot,
                            'x_p': x_pm,
                            'x_e': x_e,
                            'f_sum': f_sum,
                            'delta_sum': delta_sum,
                            'b_sum': b_sum,
                            'k_sum': k_sum
                        }
                        return solution
        
        except Exception as e:
            print(f"Ошибка расчета для {powder1}+{powder2}: {e}")
            
    except Exception as e:
        print(f"Ошибка в run_simulation: {e}")
    
    return None

def process_results(result, successful_solutions, pbar):
    """Обрабатывает результаты расчета"""
    pbar.update(1)
    
    if result is not None:
        successful_solutions.append(result)
        save_solution_to_file(result)
        
        # print(f"    УСПЕХ!!!!!!!!: {result['mixture_name']}")
        # print(f"   Состав: {result['powder1']} + {result['powder2']}")
        # print(f"   α={result['alpha']:.2f}, ω={result['omega_sum']:.2f}кг, ω/q={result['omega_q_ratio']:.2f}")
        # print(f"   Z_b1={result['Z_b1']} | V={result['velocity']:.1f}м/с | Pmax={result['pressure_max']/1e6:.1f}МПа")
        # print(f"   V(-50°C)={result['velocity_cold']:.1f}м/с | P(+50°C)={result['pressure_hot']/1e6:.1f}МПа")

def calculation_2_multiprocess(db, omega_sum_range, delta_range, alpha_range):
    """Многопроцессорная версия расчета"""
    
    initialize_result_file()

    powder_combinations = list(itertools.combinations(db, 2))
    
    # Создаем все комбинации параметров
    all_params = []
    for powder1, powder2 in powder_combinations:
        for omega_sum in omega_sum_range:
            for delta in delta_range:
                for alpha in alpha_range:
                    all_params.append((powder1, powder2, omega_sum, delta, alpha, base_opts, powders_DB))
    
    total_iterations = len(all_params)
    print(f"Всего комбинаций для перебора: {total_iterations}")
    
    # Засекаем время начала расчета
    start_time = time.time()
    
    # Создаем менеджер для общих переменных
    with mp.Manager() as manager:
        successful_solutions = manager.list()
        
        # Создаем прогресс-бар
        with tqdm(total=total_iterations, desc="Выполнение расчетов") as pbar:
            # Создаем пул процессов
            num_processes = min(cpu_count(), 8)  # Ограничиваем количество процессов
            print(f"Используется {num_processes} процессов для расчета")
            
            with Pool(processes=num_processes) as pool:
                # Используем imap_unordered для обработки результатов по мере их поступления
                results = pool.imap_unordered(run_simulation, all_params, chunksize=10)
                
                # Обрабатываем результаты
                for result in results:
                    process_results(result, successful_solutions, pbar)
        
        # Преобразуем в обычный список
        successful_solutions_list = list(successful_solutions)
    
    # Выводим итоговый результат
    elapsed = time.time() - start_time
    print(f"\nРасчет завершен за {elapsed:.2f} секунд")
    print(f"\n{'='*60}")
    print(f"НАЙДЕНО УСПЕШНЫХ РЕШЕНИЙ: {len(successful_solutions_list)}")
    print(f"{'='*60}")
    
    if successful_solutions_list:
        successful_solutions_list.sort(key=lambda x: x['Z_b1'])
        
        print("\nЛучшие решения:")
        for i, sol in enumerate(successful_solutions_list[:10]):
            print(f"{i+1}. {sol['mixture_name']}")
            print(f"   Состав: {sol['powder1']} + {sol['powder2']}")
            print(f"   α={sol['alpha']:.2f}, ω/q={sol['omega_q_ratio']:.2f}")
            print(f"   Массы: {sol['mass1']:.2f} + {sol['mass2']:.2f} = {sol['omega_sum']:.2f} кг")
            print(f"   V={sol['velocity']:.1f} м/с | Pmax={sol['pressure_max']/1e6:.1f} МПа")
            print(f"   V(-50°C)={sol['velocity_cold']:.1f} м/с | P(+50°C)={sol['pressure_hot']/1e6:.1f} МПа")
            print(f"   Z_b1={sol['Z_b1']:.2e}")
            print()
    
    return successful_solutions_list

# Запуск многопроцессорной версии
if __name__ == "__main__":
    # Устанавливаем метод запуска процессов
    # mp.set_start_method('spawn', force=True)
    # solutions = calculation_2_multiprocess(db, omega_sum_range, delta_range, alpha_range)
    print(db)
    print(len(db))