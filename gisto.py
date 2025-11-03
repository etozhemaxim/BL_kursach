from pyballistics import ozvb_termo, get_options_sample, get_db_powder
import numpy as np
import random

def get_powder_data(powder_name):
    """Получает данные пороха через get_db_powder"""
    try:
        return get_db_powder(powder_name)
    except Exception as e:
        print(f"Ошибка получения данных для пороха {powder_name}: {e}")
        return None

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
    q =5
    K = 1
    v_p = result['v_p'][-1]
    x_p = result['x_p'][-1]

    print("\n" + "="*60)
    print("РАСЧЕТ КРИТЕРИЯ СЛУХОЦКОГО Z_b1")
    print("="*60)
    
    # 1. Расчет потенциала пороха Pi
    Pi = f_sum / (k - 1)
    print(f"1. Потенциал пороха Pi = f_sum / (k - 1) = {f_sum} / ({k} - 1) = {Pi:.2f}")
    
    # 2. Расчет коэффициента фиктивной массы phi
    phi = phi_1 + (1 / (3 * q)) * (omega_ign + omega_sum)
    print(f"2. Коэффициент фиктивной массы phi = phi_1 + (1/(3*q))*(omega_ign + omega_sum) = {phi_1} + (1/(3*{q}))*({omega_ign} + {omega_sum}) = {phi:.4f}")
    
    # 3. Расчет дульной энергии E_m
    E_m = (q * v_p**2) / 2
    print(f"3. Дульная энергия E_m = (q * v_p²) / 2 = ({q} * {v_p}²) / 2 = {E_m:.2f} Дж")
    
    # 4. Расчет термического КПД eta_rm
    eta_rm = phi * E_m / Pi
    print(f"4. Термический КПД eta_rm = phi * E_m / Pi = {phi:.4f} * {E_m:.2f} / {Pi:.2f} = {eta_rm:.6f}")
    
    # 5. Расчет коэффициента использования eta_re
    eta_re = E_m / x_p
    print(f"5. Коэффициент использования eta_re = E_m / omega_sum = {E_m:.2f} / {omega_sum} = {eta_re:.6f}")
    
    # 6. Расчет приведенного пути Lambda_e
    Lambda_e = x_e / x_p
    print(f"6. Приведенный путь Lambda_e = x_e / x_p = {x_e} / {x_p} = {Lambda_e:.6f}")
    
    # 7. Расчет параметра zeta
    term1 = (1 / vardelta) - (1 / delta)
    term2 = 1 + ((b * p_ign) / f_sum)
    zeta = (p_ign / f_sum) * term1 * (1 / term2)
    print(f"7. Параметр zeta = (p_ign/f_sum) * ((1/vardelta) - (1/delta)) * (1/(1 + (b*p_ign/f_sum)))")
    print(f"   = ({p_ign}/{f_sum}) * ((1/{vardelta}) - (1/{delta})) * (1/(1 + ({b}*{p_ign}/{f_sum})))")
    print(f"   = {zeta:.8f}")
    
    # 8. Расчет приведенного пути Lamda_m
    term_A = 1 - b * vardelta * (1 + zeta) + Lambda_e
    term_B_numerator = 1 + zeta + eta_re
    term_B_denominator = 1 + zeta - eta_rm
    term_B = term_B_numerator / term_B_denominator
    exponent = 1 / (k - 1)
    term_C = term_B ** exponent
    term_D = 1 - b * vardelta * (1 + zeta)
    
    Lamda_m = term_A * term_C - term_D
    
    print(f"8. Приведенный путь Lamda_m = (1 - b*vardelta*(1+zeta) + Lambda_e) * ((1+zeta-eta_re)/(1+zeta-eta_rm))^(1/(k-1)) - (1 - b*vardelta*(1+zeta))")
    print(f"   = (1 - {b}*{vardelta}*(1+{zeta:.6f}) + {Lambda_e:.6f}) * ((1+{zeta:.6f}-{eta_re:.6f})/(1+{zeta:.6f}-{eta_rm:.6f}))^(1/({k}-1)) - (1 - {b}*{vardelta}*(1+{zeta:.6f}))")
    print(f"   = {term_A:.6f} * ({term_B_numerator:.6f}/{term_B_denominator:.6f})^({exponent:.6f}) - {term_D:.6f}")
    print(f"   = {term_A:.6f} * {term_B:.6f}^{exponent:.6f} - {term_D:.6f}")
    print(f"   = {term_A:.6f} * {term_C:.6f} - {term_D:.6f}")
    print(f"   = {Lamda_m:.6f}")
    
    # 9. Расчет показателя живучести N_s
    N_s = K * (1 + Lamda_m) * (q / omega_sum)
    print(f"9. Показатель живучести N_s = K * (1 + Lamda_m) * (q / omega_sum) = {K} * (1 + {Lamda_m:.6f}) * ({q} / {omega_sum}) = {N_s:.6f}")
    
    # 10. Расчет критерия Слухоцкого Z_b1
    term_E = (q * v_p**2 / 2)**5
    term_F = np.sqrt(N_s) / (omega_sum * x_p**4)
    Z_b1 = term_E * term_F
    
    print(f"10. Критерий Слухоцкого Z_b1 = (q * v_p² / 2)^5 * (√N_s / (omega_sum * x_p⁴))")
    print(f"    = ({q} * {v_p}² / 2)^5 * (√{N_s:.6f} / ({omega_sum} * {x_p}⁴))")
    print(f"    = {term_E:.2e} * {term_F:.2e}")
    print(f"    = {Z_b1:.2e}")
    
    print("="*60)
    
    return Z_b1

def single_calculation():
    """Единичный расчет смеси двух случайных порохов"""
    
    # Список доступных порохов
    powders_list = ['ДГ-4 14/1', '4/1 фл', '5/1', 'ДГ-3 13/1', 'ДГ-4 15/1', 
                   'АПЦ-235 П 16/1', 'ДГ-3 14/1', 'МАП-1 23/1', 'БНГ-1355 25/1']
    
    # Выбираем два случайных пороха
    powder1, powder2 = random.sample(powders_list, 2)
    
    print("ЕДИНИЧНЫЙ РАСЧЕТ СМЕСИ ДВУХ ПОРОХОВ")
    print("="*50)
    print(f"Порох 1: {powder1}")
    print(f"Порох 2: {powder2}")
    
    # Получаем данные порохов
    powder1_data = get_powder_data(powder1)
    powder2_data = get_powder_data(powder2)
    
    if powder1_data is None or powder2_data is None:
        print("Ошибка: не удалось получить данные порохов")
        return
    
    # Параметры расчета
    opts = get_options_sample()
    q = opts['init_conditions']['q'] = 5
    d = opts['init_conditions']['d'] = 0.085
    phi_1 = opts['init_conditions']['phi_1'] = 1.04
    p_0 = opts['init_conditions']['p_0'] = 30000000.0
    opts['stop_conditions']['v_p'] = 1500
    opts['stop_conditions']['p_max'] = 390000000.0
    opts['stop_conditions']['x_p'] = 5.625
    
    # Параметры смеси
    omega_sum = 3.0  # кг
    delta = 600      # кг/м³
    alpha = 0.5      # доля первого пороха
    q = 5 
    
    print(f"\nПАРАМЕТРЫ РАСЧЕТА:")
    print(f"Общая масса пороха omega_sum = {omega_sum} кг")
    print(f"Плотность заряжания delta = {delta} кг/м³")
    print(f"Доля первого пороха alpha = {alpha}")
    
    # Вычисляем массы порохов
    mass1 = omega_sum * alpha
    mass2 = omega_sum * (1 - alpha)
    W_0 = omega_sum / delta
    
    print(f"Масса пороха 1: {mass1:.2f} кг")
    print(f"Масса пороха 2: {mass2:.2f} кг")
    print(f"Объем камеры W_0 = {W_0:.4f} м³")
    
    # Настраиваем опции
    opts['powders'] = [
        {'omega': mass1, 'dbname': powder1},
        {'omega': mass2, 'dbname': powder2}
    ]
    opts['init_conditions']['W_0'] = W_0
    
    # Рассчитываем средние параметры смеси
    f_sum = alpha * powder1_data['f'] + (1 - alpha) * powder2_data['f']
    delta_sum = alpha * powder1_data['delta'] + (1 - alpha) * powder2_data['delta']
    b_sum = alpha * powder1_data['b'] + (1 - alpha) * powder2_data['b']
    k_sum = alpha * powder1_data['k'] + (1 - alpha) * powder2_data['k']
    
    print(f"\nПАРАМЕТРЫ СМЕСИ:")
    print(f"Суммарная сила пороха f_sum = {f_sum:.0f} Дж/кг")
    print(f"Суммарная плотность delta_sum = {delta_sum:.1f} кг/м³")
    print(f"Суммарный коволюм b_sum = {b_sum:.6f}")
    print(f"Суммарный показатель адиабаты k_sum = {k_sum:.3f}")
    
    try:
        # Расчет прямой задачи
        result = ozvb_termo(opts)
        
        # Основные результаты
        x_pm = result['x_p'][-1]
        v_pm = result['v_p'][-1]
        p_max = max(result['p_m'])
        x_e = x_e_func(result)
        
        print(f"\nРЕЗУЛЬТАТЫ РАСЧЕТА ПРЯМОЙ ЗАДАЧИ:")
        print(f"Дульная скорость v_p = {v_pm:.1f} м/с")
        print(f"Максимальное давление p_max = {p_max/1e6:.1f} МПа")
        print(f"Путь снаряда x_p = {x_pm:.3f} м")
        print(f"Координата конца горения x_e = {x_e:.3f} м")
        
        # Расчет критерия Слухоцкого
        Z_b1 = Z_b1_func(x_e, f_sum, omega_sum, delta, delta_sum, b_sum, k_sum, result)
        
        print(f"\nИТОГОВЫЙ РЕЗУЛЬТАТ:")
        print(f"Критерий Слухоцкого Z_b1 = {Z_b1:.2e}")
        
    except Exception as e:
        print(f"Ошибка расчета: {e}")

# Запуск единичного расчета
if __name__ == "__main__":
    single_calculation()