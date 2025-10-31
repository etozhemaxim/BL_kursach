from pyballistics import ozvb_termo, get_options_sample
import numpy as np
from tqdm import tqdm
import itertools

from baza_powder import powders_db

# Параметры для варьирования
omega_sum_range = np.linspace(0.1, 8, 4)  # общая масса пороха в кг
delta_range = np.linspace(300, 700, 8)   # плотность заряжания
alpha_range = np.linspace(0.1, 0.9, 14)    # доля первого пороха (0.1-0.9)


class Powders:

    def __init__(self):
        self.powders = ['ДГ-4 14/1', '4/1 фл', '5/1', 'ДГ-3 13/1', '4/1 фл', '5/1', 'ДГ-4 15/1', '4/1 фл', '5/1', 'АПЦ-235 П 16/1', '4/1 фл', '5/1', 'ДГ-3 14/1', '4/1 фл', '5/1', 'МАП-1 23/1', '4/1 фл', '5/1', 'БНГ-1355 25/1', '4/1 фл', '5/1', 'НДТ-3 16/1', '4/1 фл', '5/1', 'ДГ-2 15/1', '4/1 фл', '5/1', 'УГФ-1', '4/1 фл', '5/1', 'УГ-1', '4/1 фл', '5/1', 'ДГ-3 17/1', '4/1 фл', '5/1', 'НДТ-2 16/1', '4/1 фл', '5/1', 'НДТ-3 18/1', '4/1 фл', '5/1', 'ДГ-3 18/1', '4/1 фл', '5/1', 'ДГ-2 17/1', '4/1 фл', '5/1', 'НДТ-3 19/1', '4/1 фл', '5/1', 'ДГ-3 20/1', '4/1 фл', '5/1', 'НДТ-2 19/1', '4/1 фл', '5/1', '12/1 тр МН', '4/1 фл', '5/1', '7/1 УГ', '4/1 фл', '5/1', '15/1 тр В/А', '4/1 фл', '5/1', '8/1 УГ', '4/1 фл', '5/1', '16/1 тр В/А', '4/1 фл', '5/1', '11/1 БП', '4/1 фл', '5/1', '12/1 тр БП', '4/1 фл', '5/1', '18/1 тр', '4/1 фл', '5/1', '16/1 тр', '4/1 фл', '5/1', '22/1 тр', '4/1 фл', '5/1', '11/1 УГ', '4/1 фл', '5/1', '12/1 УГ', '4/1 фл', '5/1', '18/1 тр БП', '4/1 фл', '5/1', '9/7 МН', '4/1 фл', '5/1', '12/7', '4/1 фл', '5/1', '14/7 В/А', '4/1 фл', '5/1', '15/7', '4/1 фл', '5/1', '9/7 БП', '4/1 фл', '5/1', '14/7', '4/1 фл', '5/1', '17/7', '4/1 фл', '5/1', '14/7 БП', '4/1 фл', '5/1']
    
powders_DB = Powders()
available_powders = [p for p in powders_DB.powders[35:40] if p in powders_db]
db = available_powders


opts = get_options_sample()
q = opts['init_conditions']['q'] = 5
d = opts['init_conditions']['d'] = 0.085
phi_1 = opts['init_conditions']['phi_1'] = 1.04
p_0 = opts['init_conditions']['p_0'] = 30000000.0
opts['stop_conditions']['v_p'] = 1500
opts['stop_conditions']['p_max'] = 390000000.0
opts['stop_conditions']['x_p'] = 5.625

# Создаем список для успешных решений
successful_solutions = []

def save_solution_to_file(solution, filename="result2.txt"):
    """Сохраняет решение в файл"""
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"{solution['powder1']},{solution['powder2']},"
                f"{solution['mass1']:.3f},{solution['mass2']:.3f},"
                f"{solution['alpha']:.3f},{solution['omega_sum']:.3f},"
                f"{solution['omega_q_ratio']:.3f},{solution['delta']:.1f},"
                f"{solution['W_0']:.6f},{solution['Z_b1']:.6e},"
                f"{solution['velocity']:.2f},{solution['velocity_cold']:.2f},{solution['velocity_hot']:.2f},"
                f"{solution['pressure_max']/1e6:.2f},{solution['pressure_hot']/1e6:.2f},"
                f"{solution['x_p']:.3f},{solution['x_e']:.3f},"
                f"{solution['f_sum']:.0f},{solution['delta_sum']:.1f},{solution['b_sum']:.6f},{solution['k_sum']:.3f}\n")

def initialize_result_file(filename="result2.txt"):
    """Инициализирует файл с заголовками"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write("powder1,powder2,mass1_kg,mass2_kg,alpha,omega_sum_kg,omega_q_ratio,delta_kg_m3,W0_m3,Z_b1,"
                "velocity_mps,velocity_cold_mps,velocity_hot_mps,pressure_max_mpa,pressure_hot_mpa,"
                "x_p_m,x_e_m,f_sum,delta_sum,b_sum,k_sum\n")

def calculation_2(db, omega_sum_range, delta_range, alpha_range):
    
    # Инициализируем файл результатов
    initialize_result_file()
    
    # Создаем все уникальные пары порохов
    powder_combinations = list(itertools.combinations(db, 2))
    
    # Считаем общее количество итераций для прогресс-бара
    total_iterations = len(powder_combinations) * len(omega_sum_range) * len(delta_range) * len(alpha_range)
    
    # Создаем прогресс-бар
    with tqdm(total=total_iterations, desc="Расчет смесей") as pbar:

        for powder1, powder2 in powder_combinations:
            for omega_sum in omega_sum_range:
                # Проверяем отношение ω/q
                if omega_sum / q < 0.1 or omega_sum / q > 0.8:
                    pbar.update(len(delta_range) * len(alpha_range))
                    continue
                    
                for delta in delta_range:
                    W_0 = omega_sum / delta

                    if W_0 <= 0:
                        pbar.update(len(alpha_range))
                        continue
                        
                    for alpha in alpha_range:
                        # Проверяем долю пороха
                        if alpha < 0.1 or alpha > 0.9:
                            pbar.update(1)
                            continue

                        # Вычисляем массы порохов через alpha
                        mass1 = omega_sum * alpha
                        mass2 = omega_sum * (1 - alpha)
                        
                        opts['powders'] = [
                            {'omega': mass1, 'dbname': powder1},
                            {'omega': mass2, 'dbname': powder2}
                        ]

                        powder1_data = powders_db[powder1]
                        powder2_data = powders_db[powder2]

                        fraction1 = mass1 / omega_sum
                        fraction2 = mass2 / omega_sum

                        f_sum = fraction1 * powder1_data['f'] + fraction2 * powder2_data['f']
                        delta_sum = fraction1 * powder1_data['delta'] + fraction2 * powder2_data['delta']
                        b_sum = fraction1 * powder1_data['b'] + fraction2 * powder2_data['b']
                        k_sum = fraction1 * powder1_data['k'] + fraction2 * powder2_data['k']

                        opts['init_conditions']['W_0'] = W_0

                        try:
                            # Расчет при нормальных условиях
                            result_normal = ozvb_termo(opts)
                            
                            x_pm = result_normal['x_p'][-1]
                            v_pm = result_normal['v_p'][-1]
                            p_max = max(result_normal['p_m'])

                            # Базовые ограничения
                            base_conditions_met = (p_max <= 390000000.0 and v_pm >= 950 and x_pm >= 1.5 and x_pm < 7)
                            
                            if base_conditions_met:
                                # ПЕРВАЯ ДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА: -50°C
                                opts['stop_conditions']['T_0'] = 223.15  # -50°C в Кельвинах
                                result_cold = ozvb_termo(opts)
                                v_pm_cold = result_cold['v_p'][-1]
                                
                                if v_pm_cold >= 830:
                                    # ВТОРАЯ ДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА: +50°C
                                    opts['stop_conditions']['T_0'] = 323.15  # +50°C в Кельвинах
                                    result_hot = ozvb_termo(opts)
                                    p_mz_hot = result_hot['p_m'][-1]
                                    
                                    if p_mz_hot >= 180000000:  # 180 МПа в Па
                                        # Все условия выполнены - сохраняем решение
                                        x_e = x_e_func(result_normal)
                                        Z_b1 = Z_b1_func(x_e, f_sum, omega_sum, delta, delta_sum, b_sum, k_sum, result_normal)
                                        
                                        # Сохраняем успешное решение
                                        solution = {
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
                                        successful_solutions.append(solution)
                                        
                                        # Сохраняем в файл
                                        save_solution_to_file(solution)
                                        
                                        # Обновляем описание прогресс-бара
                                        pbar.set_postfix({
                                            'Смесь': f'{powder1}+{powder2}',
                                            'Z_b1': f'{Z_b1:.2e}',
                                            'V': f'{v_pm:.0f} м/с',
                                            'V(-50)': f'{v_pm_cold:.0f} м/с',
                                            'P(+50)': f'{p_mz_hot/1e6:.0f} МПа'
                                        })
                                        
                                        print(f" ✓ УСПЕХ: {powder1} + {powder2}")
                                        print(f"   α={alpha:.2f}, ω={omega_sum:.2f}кг, ω/q={omega_sum/q:.2f}")
                                        print(f"   Z_b1={Z_b1:.2e} | V={v_pm:.1f}м/с | Pmax={p_max/1e6:.1f}МПа")
                                        print(f"   V(-50°C)={v_pm_cold:.1f}м/с | P(+50°C)={p_mz_hot/1e6:.1f}МПа")
                                    else:
                                        # Не прошла проверка при +50°C
                                        pbar.set_postfix({
                                            'Смесь': f'{powder1[:5]}+{powder2[:5]}',
                                            'Статус': '✗ +50°C',
                                            'P(+50)': f'{p_mz_hot/1e6:.0f} МПа'
                                        })
                                else:
                                    # Не прошла проверка при -50°C
                                    pbar.set_postfix({
                                        'Смесь': f'{powder1[:5]}+{powder2[:5]}',
                                        'Статус': '✗ -50°C',
                                        'V(-50)': f'{v_pm_cold:.0f} м/с'
                                    })
                            else:
                                # Не прошла базовая проверка
                                pbar.set_postfix({
                                    'Смесь': f'{powder1[:5]}+{powder2[:5]}',
                                    'V': f'{v_pm:.0f} м/с',
                                    'P': f'{p_max/1e6:.0f} МПа'
                                })
                                print(f" {powder1} + {powder2} | α={alpha:.2f} | V={v_pm:.1f}м/с | P={p_max/1e6:.1f}МПа| L={x_pm:.3f}м |")
                                
                        except Exception as e:
                            print(f" Ошибка расчета для {powder1}+{powder2}: {e}")
                        
                        pbar.update(1)
        
        # Выводим итоги
        print(f"\n{'='*60}")
        print(f"НАЙДЕНО УСПЕШНЫХ РЕШЕНИЙ: {len(successful_solutions)}")
        print(f"Диапазон α: {alpha_range[0]:.1f}...{alpha_range[-1]:.1f}")
        print(f"Диапазон ω/q: {omega_sum_range[0]/q:.1f}...{omega_sum_range[-1]/q:.1f}")
        print(f"Результаты сохранены в файл: result2.txt")
        print(f"{'='*60}")
        
        if successful_solutions:
            # Сортируем по Z_b1 (чем меньше, тем лучше)
            successful_solutions.sort(key=lambda x: x['Z_b1'])
            print("Лучшие решения (по Z_b1):")
            for i, sol in enumerate(successful_solutions[:10]):
                print(f"{i+1}. {sol['powder1']} + {sol['powder2']}")
                print(f"   α={sol['alpha']:.2f}, ω/q={sol['omega_q_ratio']:.2f}")
                print(f"   Массы: {sol['mass1']:.2f} + {sol['mass2']:.2f} = {sol['omega_sum']:.2f} кг")
                print(f"   V={sol['velocity']:.1f} м/с | Pmax={sol['pressure_max']/1e6:.1f} МПа")
                print(f"   V(-50°C)={sol['velocity_cold']:.1f} м/с | P(+50°C)={sol['pressure_hot']/1e6:.1f} МПа")
                print(f"   Z_b1={sol['Z_b1']:.2e}")
                print()
        
        return successful_solutions


def Z_b1_func(x_e, f_sum, omega_sum, vardelta, delta, b, k, result):
    phi_1 = 1.04
    p_ign = 5e6
    omega_ign = 0.01
    K = 0.5
    v_p = result['v_p'][-1]
    x_p = result['x_p'][-1]

    Pi = f_sum / (k - 1) 
    phi = phi_1 * (1 / (3 * q)) * (omega_ign * omega_sum)   
    E_m = (q * v_p**2) / 2
    eta_rm = phi * E_m / Pi    
    eta_re = E_m / omega_sum
    Lambda_e = x_e / x_p

    zeta = (p_ign / f_sum) * ((1 / vardelta) - (1 / delta)) * (1 / (1 + ((b * p_ign) / f_sum)))
    Lamda_m = (1 - b * vardelta * (1 + zeta) + Lambda_e) * ((1 + zeta - eta_re) / (1 + zeta - eta_rm))**(1 / (k - 1)) - (1 - b * vardelta * (1 + zeta))
    N_s = K * (1 + Lamda_m) * (q / omega_sum)
    Z_b1 = (q * v_p**2 / 2)**5 * (np.sqrt(N_s) / omega_sum * x_p)

    return Z_b1


def x_e_func(result):
    psi_1 = result['psi_1'] 
    psi_2 = result['psi_2'] 
    x_p = result['x_p']
    
    tolerance = 1e-3
    
    for i in range(len(psi_1)):
        if (abs(psi_1[i] - 1.0) < tolerance and abs(psi_2[i] - 1.0) < tolerance):
            return x_p[i]

    return x_p[-1]


# Запускаем расчет
solutions = calculation_2(db, omega_sum_range, delta_range, alpha_range)