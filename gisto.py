from pyballistics import ozvb_lagrange, get_options_sample
import numpy as np
from tqdm import tqdm
import itertools
from baza_powder import powders_db

# Отношение массы пороха к массе снаряда (0.1...0.8)
omega_q_ratio = np.linspace(0.1, 0.8, 5)
vardelta_db = np.linspace(500, 1200, 5)
alpha_range = np.linspace(0.1, 0.9, 5)    # доля первого пороха (0.1-0.9)

class Powders:

    def __init__(self):
        self.powders = ['ДГ-4 14/1', '4/1 фл', '5/1', 'ДГ-3 13/1', '4/1 фл', '5/1', 'ДГ-4 15/1', '4/1 фл', '5/1', 'АПЦ-235 П 16/1', '4/1 фл', '5/1', 'ДГ-3 14/1', '4/1 фл', '5/1', 'МАП-1 23/1', '4/1 фл', '5/1', 'БНГ-1355 25/1', '4/1 фл', '5/1', 'НДТ-3 16/1', '4/1 фл', '5/1', 'ДГ-2 15/1', '4/1 фл', '5/1', 'УГФ-1', '4/1 фл', '5/1', 'УГ-1', '4/1 фл', '5/1', 'ДГ-3 17/1', '4/1 фл', '5/1', 'НДТ-2 16/1', '4/1 фл', '5/1', 'НДТ-3 18/1', '4/1 фл', '5/1', 'ДГ-3 18/1', '4/1 фл', '5/1', 'ДГ-2 17/1', '4/1 фл', '5/1', 'НДТ-3 19/1', '4/1 фл', '5/1', 'ДГ-3 20/1', '4/1 фл', '5/1', 'НДТ-2 19/1', '4/1 фл', '5/1', '12/1 тр МН', '4/1 фл', '5/1', '7/1 УГ', '4/1 фл', '5/1', '15/1 тр В/А', '4/1 фл', '5/1', '8/1 УГ', '4/1 фл', '5/1', '16/1 тр В/А', '4/1 фл', '5/1', '11/1 БП', '4/1 фл', '5/1', '12/1 тр БП', '4/1 фл', '5/1', '18/1 тр', '4/1 фл', '5/1', '16/1 тр', '4/1 фл', '5/1', '22/1 тр', '4/1 фл', '5/1', '11/1 УГ', '4/1 фл', '5/1', '12/1 УГ', '4/1 фл', '5/1', '18/1 тр БП', '4/1 фл', '5/1', '9/7 МН', '4/1 фл', '5/1', '12/7', '4/1 фл', '5/1', '14/7 В/А', '4/1 фл', '5/1', '15/7', '4/1 фл', '5/1', '9/7 БП', '4/1 фл', '5/1', '14/7', '4/1 фл', '5/1', '17/7', '4/1 фл', '5/1', '14/7 БП', '4/1 фл', '5/1']
    
powders_DB = Powders()
available_powders = [p for p in powders_DB.powders[20:30] if p in powders_db]
db = available_powders


opts = get_options_sample()
q = opts['init_conditions']['q'] = 5  # масса снаряда
d = opts['init_conditions']['d'] = 0.085
phi_1 = opts['init_conditions']['phi_1'] = 1.04
p_0 = opts['init_conditions']['p_0'] = 30000000.0
opts['stop_conditions']['v_p'] = 1500
opts['stop_conditions']['p_max'] = 390000000.0
opts['stop_conditions']['x_p'] = 5.625

# Создаем список для успешных решений
successful_solutions = []

def calculation_2(db, omega_q_ratio, vardelta_db):
    
    # Создаем все уникальные пары порохов
    powder_combinations = list(itertools.combinations(db, 2))
    
    # Считаем общее количество итераций для прогресс-бара
    total_iterations = len(powder_combinations) * len(omega_q_ratio) * len(vardelta_db) * len(alpha_range)
    
    # Создаем прогресс-бар
    with tqdm(total=total_iterations, desc="Расчет смесей") as pbar:

        for powder1, powder2 in powder_combinations:
            for omega_sum in omega_q_ratio:
                # Проверяем отношение ω/q
                if omega_sum / q < 0.1 or omega_sum / q > 0.8:
                    pbar.update(len(vardelta_db) * len(alpha_range))
                    continue
                    
                for vardelta in vardelta_db:
                    W_0 = omega_sum / vardelta

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
                            # Расчет ГД модели
                            result = ozvb_lagrange(opts)
                            
                            # Извлекаем данные из ГД модели
                            # Максимальное давление - максимальное значение из всех ячеек во всех временных слоях
                            p_max = max([max(layer['p']) for layer in result['layers']])
                            
                            # Дульная скорость - скорость последнего узла в последнем временном слое
                            last_layer = result['layers'][-1]
                            v_muzzle = last_layer['u'][-1]  # скорость последнего узла
                            
                            # Длина ствола - координата последнего узла
                            barrel_length = last_layer['x'][-1]
                            
                            # Дополнительно: путь снаряда (если есть в результатах)
                            x_pm = result['x_p'][-1] if 'x_p' in result else barrel_length

                            # Проверяем базовые условия
                            base_conditions_met = (p_max <= 390000000.0 and v_muzzle >= 950 and x_pm >= 1.5 and x_pm < 5.525)
                            
                            if base_conditions_met:
                                # Проверка при -50°C
                                opts['stop_conditions']['T_0'] = 223.15
                                result_cold = ozvb_lagrange(opts)
                                last_layer_cold = result_cold['layers'][-1]
                                v_muzzle_cold = last_layer_cold['u'][-1]
                                
                                if v_muzzle_cold >= 830:
                                    # Проверка при +50°C
                                    opts['stop_conditions']['T_0'] = 323.15
                                    result_hot = ozvb_lagrange(opts)
                                    last_layer_hot = result_hot['layers'][-1]
                                    p_mz_hot = last_layer_hot['p'][-1]  # давление в дульном срезе
                                    
                                    if p_mz_hot >= 180000000:  # 180 МПа в Па
                                        # Все условия выполнены - сохраняем решение
                                        x_e = x_e_func(result)
                                        Z_b1 = Z_b1_func(x_e, f_sum, omega_sum, vardelta, delta_sum, b_sum, k_sum, result)
                                        
                                        # Сохраняем успешное решение
                                        solution = {
                                            'powder1': powder1,
                                            'powder2': powder2,
                                            'mass1': mass1,
                                            'mass2': mass2,
                                            'alpha': alpha,
                                            'omega_sum': omega_sum,
                                            'omega_q_ratio': omega_sum / q,
                                            'vardelta': vardelta,
                                            'Z_b1': Z_b1,
                                            'velocity_muzzle': v_muzzle,
                                            'velocity_cold': v_muzzle_cold,
                                            'pressure_max': p_max,
                                            'pressure_muzzle_hot': p_mz_hot,
                                            'barrel_length': barrel_length,
                                            'x_p': x_pm,
                                            'x_e': x_e,
                                            'f_sum': f_sum,
                                            'delta_sum': delta_sum,
                                            'b_sum': b_sum,
                                            'k_sum': k_sum
                                        }
                                        successful_solutions.append(solution)
                                        
                                        # Обновляем описание прогресс-бара
                                        pbar.set_postfix({
                                            'Смесь': f'{powder1[:5]}+{powder2[:5]}',
                                            'Z_b1': f'{Z_b1:.2e}',
                                            'V_дульн': f'{v_muzzle:.0f} м/с',
                                            'P_max': f'{p_max/1e6:.0f} МПа',
                                            'Условия': '✓ ВСЕ'
                                        })
                                        
                                        print(f" ✓ УСПЕХ: {powder1} + {powder2}")
                                        print(f"   α={alpha:.2f}, ω={omega_sum:.2f}кг")
                                        print(f"   Z_b1={Z_b1:.2e} | V_дульн={v_muzzle:.1f}м/с | P_max={p_max/1e6:.1f}МПа")
                                        print(f"   Длина ствола: {barrel_length:.3f} м")
                                        print(f"   V(-50°C)={v_muzzle_cold:.1f}м/с | P(+50°C)={p_mz_hot/1e6:.1f}МПа")
                                        
                            else:
                                # Обновляем описание прогресс-бара для неуспешных решений
                                pbar.set_postfix({
                                    'Смесь': f'{powder1[:5]}+{powder2[:5]}',
                                    'V_дульн': f'{v_muzzle:.0f} м/с',
                                    'P_max': f'{p_max/1e6:.0f} МПа',
                                    'Условия': '✗ БАЗА'
                                })
                                print(f" {powder1} + {powder2} | V={v_muzzle:.1f}м/с | P={p_max/1e6:.1f}МПа")
                                
                        except Exception as e:
                            print(f" Ошибка расчета для {powder1}+{powder2}: {e}")
                        
                        pbar.update(1)
        
        # Выводим итоги
        print(f"\n{'='*60}")
        print(f"НАЙДЕНО УСПЕШНЫХ РЕШЕНИЙ: {len(successful_solutions)}")
        print(f"{'='*60}")
        
        if successful_solutions:
            # Сортируем по Z_b1 (чем меньше, тем лучше)
            successful_solutions.sort(key=lambda x: x['Z_b1'])
            print("Лучшие решения (по Z_b1):")
            for i, sol in enumerate(successful_solutions[:10]):
                print(f"{i+1}. {sol['powder1']} + {sol['powder2']}")
                print(f"   α={sol['alpha']:.2f}, ω/q={sol['omega_q_ratio']:.2f}")
                print(f"   V_дульн={sol['velocity_muzzle']:.1f} м/с | P_max={sol['pressure_max']/1e6:.1f} МПа")
                print(f"   Длина ствола: {sol['barrel_length']:.3f} м")
                print(f"   Z_b1={sol['Z_b1']:.2e}")
                print()
        
        return successful_solutions


def Z_b1_func(x_e, f_sum, omega_sum, vardelta, delta, b, k, result):
    # Функция расчета Z_b1 (адаптировать под ГД модель если нужно)
    phi_1 = 1.04
    p_ign = 5e6
    omega_ign = 0.01
    K = 0.5
    
    # Берем данные из последнего временного слоя
    last_layer = result['layers'][-1]
    v_p = last_layer['u'][-1]  # дульная скорость
    x_p = last_layer['x'][-1]  # длина ствола

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
    # Функция нахождения пути прогорания заряда (адаптировать под ГД модель)
    # Для ГД модели нужно анализировать все временные слои
    last_layer = result['layers'][-1]
    psi_1 = last_layer['psi_1'] 
    psi_2 = last_layer['psi_2'] 
    x_p = last_layer['x'][-1]
    
    tolerance = 1e-3
    
    # Ищем слой, где оба пороха сгорели
    for layer in result['layers']:
        if (np.all(np.abs(layer['psi_1'] - 1.0) < tolerance) and 
            np.all(np.abs(layer['psi_2'] - 1.0) < tolerance)):
            return layer['x'][-1]  # возвращаем длину ствола в момент полного сгорания

    return x_p  # если полного сгорания не достигнуто


# Запускаем расчет
solutions = calculation_2(db, omega_q_ratio, vardelta_db)