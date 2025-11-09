from pyballistics import ozvb_termo, get_options_sample, get_db_powder, ozvb_lagrange
import numpy as np
from tqdm import tqdm
import itertools

omega_sum_range = np.linspace(4.7, 5, 100)  # общая масса пороха в кг
delta_range = np.linspace(970, 1000, 40)   # плотность заряжания
alpha_range = np.linspace(0.5, 0.7, 30)    # доля первого пороха (0.1-0.9)


class Powders:

    def __init__(self):
        self.powders = ['ДГ-4 14/1', 'ДГ-3 13/1', 'ДГ-4 15/1', 'АПЦ-235 П 16/1', 'ДГ-3 14/1', 'МАП-1 23/1', 'БНГ-1355 25/1', 'НДТ-3 16/1', 'ДГ-2 15/1', 'УГФ-1', 'УГ-1', 'ДГ-3 17/1', 'НДТ-2 16/1', 'НДТ-3 18/1', 'ДГ-3 18/1', 'ДГ-2 17/1', 'НДТ-3 19/1', 'ДГ-3 20/1', 'НДТ-2 19/1', '12/1 тр МН', '7/1 УГ', '15/1 тр В/А', '8/1 УГ', '16/1 тр В/А', '11/1 БП', '12/1 тр БП', '18/1 тр', '16/1 тр', '22/1 тр', '11/1 УГ', '12/1 УГ', '18/1 тр БП', '9/7 МН', '12/7', '14/7 В/А', '15/7', '9/7 БП', '14/7', '17/7', '14/7 БП']


def get_powder_data(powder_name):
    """Получает данные пороха через get_db_powder"""
    try:
        return get_db_powder(powder_name)
    except Exception as e:
        print(f"Ошибка получения данных для пороха {powder_name}: {e}")
        return None

powders_DB = Powders()

available_powders = []
for p in powders_DB.powders[:]:
    powder_data = get_powder_data(p)
    if powder_data is not None:
        available_powders.append(p)

db = available_powders

opts = get_options_sample()
q = opts['init_conditions']['q'] = 5
d = opts['init_conditions']['d'] = 0.085
phi_1 = opts['init_conditions']['phi_1'] = 1.04
p_0 = opts['init_conditions']['p_0'] = 30000000.0
opts['stop_conditions']['v_p'] = 950
opts['stop_conditions']['p_max'] = 390000000.0
opts['stop_conditions']['x_p'] = 5.625


successful_solutions = []

def save_solution_to_file(solution, filename="result2.txt"):
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
    with open(filename, "w", encoding="utf-8") as f:
        f.write("powder1,powder2,mass1_kg,mass2_kg,alpha,omega_sum_kg,omega_q_ratio,delta_kg_m3,W0_m3,Z_b1,"
                "velocity_mps,velocity_cold_mps,velocity_hot_mps,pressure_max_mpa,pressure_hot_mpa,"
                "x_p_m,x_e_m,f_sum,delta_sum,b_sum,k_sum\n")


def calculation_2(db, omega_sum_range, delta_range, alpha_range):

    
    initialize_result_file()

    powder_combinations = list(itertools.combinations(db, 2))
    
    total_iterations = len(powder_combinations) * len(omega_sum_range) * len(delta_range) * len(alpha_range)
    
    with tqdm(total=total_iterations, desc="Расчет смесей") as pbar:

        for powder1, powder2 in powder_combinations:

            powder1_data = get_powder_data(powder1)
            powder2_data = get_powder_data(powder2)
            
            if powder1_data is None or powder2_data is None:
                print(f"Пропускаем пару {powder1}+{powder2} - нет данных о порохах")
                pbar.update(len(omega_sum_range) * len(delta_range) * len(alpha_range))
                continue
                
            for omega_sum in omega_sum_range:

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

                        mass1 = omega_sum * alpha
                        mass2 = omega_sum * (1 - alpha)
                        
                        opts['powders'] = [
                            {'omega': mass1, 'dbname': powder1},
                            {'omega': mass2, 'dbname': powder2}
                        ]

                        fraction1 = mass1 / omega_sum
                        fraction2 = mass2 / omega_sum

                        f_sum = alpha * powder1_data['f']  + (1 - alpha) * powder2_data['f'] 
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
                            Z_b1, Pi, phi, E_m,  eta_rm,  eta_re, Lambda_e, zeta ,  Lamda_m , N_s, a, b, c = Z_b1_func(x_e, f_sum, omega_sum, delta, delta_sum, b_sum, k_sum, result_normal)
                            # print(f" {powder1} ({fraction1}) + {powder2}({fraction2}) {omega_sum} ==={Z_b1} ")
                            # print(f"  Pi = { Pi}")
                            # print(f"  phi = { phi}")
                            # print(f"  E_m = { E_m}")
                            # print(f"  eta_rm = {  eta_rm}")
                            # print(f"  eta_re = { eta_re}")
                            # print(f"  Lambda_e = { Lambda_e}")
                            # print(f"  zeta = { zeta}")
                            # print(f"  Lamda_m = { Lamda_m}")
                            # print(f"  N_s = { N_s}")
                            # print(f"  b = { b_sum}")
                            # print(f"  a = { a}")
                            # print(f"  b = { b}")
                            # print(f"  c = {c}")
                            # print(f"  k = {k_sum}")
                            # print(f"  omega = {omega_sum}")
                            # print(f"  f = {f_sum}")
                            base_conditions_met = (p_max <= 390000000.0 and v_pm >= 950 and x_pm >= 1.5 and x_pm < 5.625)
                            
                            if base_conditions_met:
                                #  -50°C
                                opts['stop_conditions']['T_0'] = 223.15  
                                result_cold = ozvb_termo(opts)
                                v_pm_cold = result_cold['v_p'][-1]
                                
                                if v_pm_cold >= 800:
                                    # +50°C
                                    opts['stop_conditions']['T_0'] = 323.15   
                                    result_hot = ozvb_termo(opts)
                                    p_mz_hot = result_hot['p_m'][-1]
                                    
                                    if p_mz_hot <= 180000000: 

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
                                        

                                        save_solution_to_file(solution)
                                        
                                        pbar.set_postfix({
                                            'Смесь': f'{powder1}+{powder2}',
                                            'Z_b1': f'{Z_b1}',
                                            'V': f'{v_pm:.0f} м/с',
                                            'V(-50)': f'{v_pm_cold:.0f} м/с',
                                            'P(+50)': f'{p_mz_hot/1e6:.0f} МПа'
                                        })
                                        
                                        print(f"    УСПЕХ!!!!!!!!: {powder1} + {powder2}")
                                        print(f"   α={alpha:.2f}, ω={omega_sum:.2f}кг, ω/q={omega_sum/q:.2f}")
                                        print(f"   Z_b1={Z_b1} | V={v_pm:.1f}м/с | Pmax={p_max/1e6:.1f}МПа")
                                        print(f"   V(-50°C)={v_pm_cold:.1f}м/с | P(+50°C)={p_mz_hot/1e6:.1f}МПа")

                            print(f" {powder1} + {powder2} | α={alpha:.2f} | V={v_pm:.1f}м/с | P={p_max/1e6:.1f}МПа | Z={Z_b1} |")
                                
                        except Exception as e:
                            print(f" Ошибка расчета для {powder1}+{powder2}: {e}")
                        
                        pbar.update(1)
        
        print(f"\n{'='*60}")
        print(f"НАЙДЕНО УСПЕШНЫХ РЕШЕНИЙ: {len(successful_solutions)}")
        print(f"{'='*60}")
        
        if successful_solutions:
            successful_solutions.sort(key=lambda x: x['Z_b1'])
            # for i, sol in enumerate(successful_solutions[:10]):
            #     print(f"{i+1}. {sol['powder1']} + {sol['powder2']}")
            #     print(f"   α={sol['alpha']:.2f}, ω/q={sol['omega_q_ratio']:.2f}")
            #     print(f"   Массы: {sol['mass1']:.2f} + {sol['mass2']:.2f} = {sol['omega_sum']:.2f} кг")
            #     print(f"   V={sol['velocity']:.1f} м/с | Pmax={sol['pressure_max']/1e6:.1f} МПа")
            #     print(f"   V(-50°C)={sol['velocity_cold']:.1f} м/с | P(+50°C)={sol['pressure_hot']/1e6:.1f} МПа")
            #     print(f"   Z_b1={sol['Z_b1']:.2e}")
            #     print()
        
        return successful_solutions
def Z_b1_func(x_e, f_sum, omega_sum, vardelta, delta, b, k, result):
    phi_1 = 1.04
    p_ign = 5e6
    omega_ign = 0.01
    K = (1e-40)**2
    v_p = result['v_p'][-1]
    x_p = result['x_p'][-1]

    Pi = f_sum / (k - 1) 

    phi = phi_1 + (1 / (3 * q)) * (omega_ign + omega_sum)   

    E_m = (q * v_p**2) / 2 

    eta_rm = phi * E_m / Pi    

    eta_re = E_m / x_p

    Lambda_e = x_e / x_p


    zeta = (p_ign / f_sum) * ((1 / vardelta) - (1 / delta)) * (1 / (1 + ((b * p_ign) / f_sum)))

    Lamda_m = (1 - b * vardelta * (1 + zeta) + Lambda_e) * ((1 + zeta + eta_re) / (1 + zeta - eta_rm))**(1 /(k-1)) - (1 - b * vardelta * (1 + zeta))

    N_s = K * (1 + Lamda_m) * (q / omega_sum)

    Z_b1 = (q * v_p**2 / 2)**5 * (np.sqrt(N_s) / omega_sum * x_p**4) 

    return Z_b1 , Pi, phi, E_m,  eta_rm,  eta_re, Lambda_e, zeta ,  Lamda_m , N_s , (1 - b * vardelta * (1 + zeta) + Lambda_e), ((1 + zeta + eta_re) / (1 + zeta - eta_rm))**(1 / (k - 1)), (1 - b * vardelta * (1 + zeta))


def x_e_func(result):
    psi_1 = result['psi_1'] 
    psi_2 = result['psi_2'] 
    x_p = result['x_p']
    
    tolerance = 1e-3
    
    for i in range(len(psi_1)):
        if (abs(psi_1[i] - 1.0) < tolerance and abs(psi_2[i] - 1.0) < tolerance):
            return x_p[i]

    return x_p[-1]

def calculation_2_lagrange(db, omega_sum_range, delta_range, alpha_range):
    
    initialize_result_file("result_lagrange.txt")

    powder_combinations = list(itertools.combinations(db, 2))
    
    total_iterations = len(powder_combinations) * len(omega_sum_range) * len(delta_range) * len(alpha_range)
    
    with tqdm(total=total_iterations, desc="Расчет смесей ГД") as pbar:

        for powder1, powder2 in powder_combinations:

            powder1_data = get_powder_data(powder1)
            powder2_data = get_powder_data(powder2)
            
            if powder1_data is None or powder2_data is None:
                print(f"Пропускаем пару {powder1}+{powder2} - нет данных о порохах")
                pbar.update(len(omega_sum_range) * len(delta_range) * len(alpha_range))
                continue
                
            for omega_sum in omega_sum_range:

                if omega_sum / q < 0.1 or omega_sum / q > 2:
                    pbar.update(len(delta_range) * len(alpha_range))
                    continue
                    
                for delta in delta_range:
                    W_0 = omega_sum / delta

                    if W_0 <= 0:
                        pbar.update(len(alpha_range))
                        continue
                        
                    for alpha in alpha_range:
                        if alpha < 0.1 or alpha > 0.9:
                            pbar.update(1)
                            continue

                        mass1 = omega_sum * alpha
                        mass2 = omega_sum * (1 - alpha)
                        
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
                            # Газодинамический расчет
                            result_lagrange = ozvb_lagrange(opts)
                            
                            # Извлекаем данные из ГД модели
                            # Максимальное давление - максимальное значение из всех ячеек во всех временных слоях
                            p_max = max([max(layer['p']) for layer in result_lagrange['layers']])
                            
                            # Дульная скорость - скорость последнего узла в последнем временном слое
                            last_layer = result_lagrange['layers'][-1]
                            v_muzzle = last_layer['u'][-1]
                            
                            # Длина ствола - координата последнего узла
                            barrel_length = last_layer['x'][-1]
                            
                            # Путь снаряда (берем из результатов если есть, иначе длину ствола)
                            x_pm = result_lagrange['x_p'][-1] if 'x_p' in result_lagrange else barrel_length

                            # Расчет Z_b1 для ГД модели
                            x_e = x_e_func_lagrange(result_lagrange)
                            Z_b1, Pi, phi, E_m, eta_rm, eta_re, Lambda_e, zeta, Lamda_m, N_s, a, b, c = Z_b1_func_lagrange(
                                x_e, f_sum, omega_sum, delta, delta_sum, b_sum, k_sum, result_lagrange)
                            
                            base_conditions_met = (p_max <= 390000000.0 and v_muzzle >= 950 and x_pm >= 1.5 and x_pm < 5.625)
                            
                            if base_conditions_met:
                                # Проверка при -50°C
                                opts['stop_conditions']['T_0'] = 223.15
                                opts['stop_conditions']['v_p'] = 830
                                result_cold = ozvb_lagrange(opts)
                                last_layer_cold = result_cold['layers'][-1]
                                v_muzzle_cold = last_layer_cold['u'][-1]
                                
                                if v_muzzle_cold >= 800:
                                    # Проверка при +50°C
                                    opts['stop_conditions']['T_0'] = 323.15
                                    result_hot = ozvb_lagrange(opts)
                                    last_layer_hot = result_hot['layers'][-1]
                                    p_mz_hot = last_layer_hot['p'][-1]  # давление в дульном срезе
                                    
                                    if p_mz_hot <= 180000000:

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
                                            'velocity': v_muzzle,
                                            'velocity_cold': v_muzzle_cold,
                                            'velocity_hot': result_hot['layers'][-1]['u'][-1],
                                            'pressure_max': p_max,
                                            'pressure_hot': p_mz_hot,
                                            'x_p': x_pm,
                                            'x_e': x_e,
                                            'barrel_length': barrel_length,
                                            'f_sum': f_sum,
                                            'delta_sum': delta_sum,
                                            'b_sum': b_sum,
                                            'k_sum': k_sum
                                        }
                                        successful_solutions.append(solution)
                                        
                                        save_solution_to_file(solution, "result_lagrange.txt")
                                        
                                        pbar.set_postfix({
                                            'Смесь': f'{powder1}+{powder2}',
                                            'Z_b1': f'{Z_b1}',
                                            'V': f'{v_muzzle:.0f} м/с',
                                            'V(-50)': f'{v_muzzle_cold:.0f} м/с',
                                            'P(+50)': f'{p_mz_hot/1e6:.0f} МПа'
                                        })
                                        
                                        print(f"    УСПЕХ ГД!!!!!!!!: {powder1} + {powder2}")
                                        print(f"   α={alpha:.2f}, ω={omega_sum:.2f}кг, ω/q={omega_sum/q:.2f}")
                                        print(f"   Z_b1={Z_b1} | V_дульн={v_muzzle:.1f}м/с | Pmax={p_max/1e6:.1f}МПа")
                                        print(f"   V(-50°C)={v_muzzle_cold:.1f}м/с | P(+50°C)={p_mz_hot/1e6:.1f}МПа")
                                        print(f"   Длина ствола: {barrel_length:.3f} м")

                            print(f" {powder1} + {powder2} | α={alpha:.2f} | V_дульн={v_muzzle:.1f}м/с | P={p_max/1e6:.1f}МПа | Z={Z_b1} |")
                                
                        except Exception as e:
                            print(f" Ошибка ГД расчета для {powder1}+{powder2}: {e}")
                        
                        pbar.update(1)
        
        print(f"\n{'='*60}")
        print(f"НАЙДЕНО УСПЕШНЫХ РЕШЕНИЙ ГД: {len(successful_solutions)}")
        print(f"{'='*60}")
        
        if successful_solutions:
            successful_solutions.sort(key=lambda x: x['Z_b1'])
        
        return successful_solutions


def Z_b1_func_lagrange(x_e, f_sum, omega_sum, vardelta, delta, b, k, result):
    """Адаптированная функция Z_b1 для ГД модели"""
    phi_1 = 1.04
    p_ign = 5e6
    omega_ign = 0.01
    K = (1e-40)**2
    
    # Берем данные из последнего временного слоя ГД модели
    last_layer = result['layers'][-1]
    v_p = last_layer['u'][-1]  # дульная скорость
    x_p = last_layer['x'][-1]  # длина ствола

    Pi = f_sum / (k - 1) 
    phi = phi_1 + (1 / (3 * q)) * (omega_ign + omega_sum)   
    E_m = (q * v_p**2) / 2 
    eta_rm = phi * E_m / Pi    
    eta_re = E_m / x_p
    Lambda_e = x_e / x_p

    zeta = (p_ign / f_sum) * ((1 / vardelta) - (1 / delta)) * (1 / (1 + ((b * p_ign) / f_sum)))
    Lamda_m = (1 - b * vardelta * (1 + zeta) + Lambda_e) * ((1 + zeta + eta_re) / (1 + zeta - eta_rm))**(1 /(k-1)) - (1 - b * vardelta * (1 + zeta))
    N_s = K * (1 + Lamda_m) * (q / omega_sum)
    Z_b1 = (q * v_p**2 / 2)**5 * (np.sqrt(N_s) / omega_sum * x_p**4) 

    return Z_b1, Pi, phi, E_m, eta_rm, eta_re, Lambda_e, zeta, Lamda_m, N_s, (1 - b * vardelta * (1 + zeta) + Lambda_e), ((1 + zeta + eta_re) / (1 + zeta - eta_rm))**(1 / (k - 1)), (1 - b * vardelta * (1 + zeta))


def x_e_func_lagrange(result):
    """Функция нахождения пути прогорания заряда для ГД модели"""
    tolerance = 1e-3
    
    # Ищем временной слой, где оба пороха полностью сгорели
    for layer in result['layers']:
        if (np.all(np.abs(layer['psi_1'] - 1.0) < tolerance) and 
            np.all(np.abs(layer['psi_2'] - 1.0) < tolerance)):
            return layer['x'][-1]  # возвращаем длину ствола в момент полного сгорания
    
    # Если полного сгорания не достигнуто, возвращаем конечную длину ствола
    return result['layers'][-1]['x'][-1]


def calculation_single_powder(db, omega_sum_range, delta_range):
    """Расчет для одного пороха"""
    
    initialize_result_file("result_single.txt")

    total_iterations = len(db) * len(omega_sum_range) * len(delta_range)
    
    with tqdm(total=total_iterations, desc="Расчет одного пороха") as pbar:

        for powder in db:
            powder_data = get_powder_data(powder)
            
            if powder_data is None:
                print(f"Пропускаем порох {powder} - нет данных")
                pbar.update(len(omega_sum_range) * len(delta_range))
                continue
                
            for omega_sum in omega_sum_range:
                if omega_sum / q < 0.1 or omega_sum / q > 0.9:
                    pbar.update(len(delta_range))
                    continue
                    
                for delta in delta_range:
                    W_0 = omega_sum / delta

                    if W_0 <= 0:
                        pbar.update(1)
                        continue

                    # Для одного пороха используем всю массу
                    mass = omega_sum
                    
                    opts['powders'] = [
                        {'omega': mass, 'dbname': powder}
                    ]

                    f_sum = powder_data['f']
                    delta_sum = powder_data['delta']
                    b_sum = powder_data['b']
                    k_sum = powder_data['k']

                    opts['init_conditions']['W_0'] = W_0

                    try:
                        # Расчет при нормальных условиях
                        result_normal = ozvb_termo(opts)
                        
                        x_pm = result_normal['x_p'][-1]
                        v_pm = result_normal['v_p'][-1]
                        p_max = max(result_normal['p_m'])

                        x_e = x_e_func(result_normal)
                        Z_b1, Pi, phi, E_m, eta_rm, eta_re, Lambda_e, zeta, Lamda_m, N_s, a, b, c = Z_b1_func(
                            x_e, f_sum, omega_sum, delta, delta_sum, b_sum, k_sum, result_normal)
                        
                        base_conditions_met = (p_max <= 390000000.0 and v_pm >= 950 and x_pm >= 1.5 and x_pm < 5.625)
                        
                        if base_conditions_met:
                            # Проверка при -50°C
                            opts['stop_conditions']['T_0'] = 223.15
                            result_cold = ozvb_termo(opts)
                            v_pm_cold = result_cold['v_p'][-1]
                            
                            if v_pm_cold >= 800:
                                # Проверка при +50°C
                                opts['stop_conditions']['T_0'] = 323.15
                                result_hot = ozvb_termo(opts)
                                p_mz_hot = result_hot['p_m'][-1]
                                
                                if p_mz_hot >= 170000000:

                                    solution = {
                                        'powder1': powder,
                                        'powder2': '',  # пусто для одного пороха
                                        'mass1': mass,
                                        'mass2': 0,
                                        'alpha': 1.0,  # 100% первого пороха
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
                                    
                                    save_solution_to_file(solution, "result_single.txt")
                                    
                                    pbar.set_postfix({
                                        'Порох': f'{powder}',
                                        'Z_b1': f'{Z_b1}',
                                        'V': f'{v_pm:.0f} м/с',
                                        'V(-50)': f'{v_pm_cold:.0f} м/с',
                                        'P(+50)': f'{p_mz_hot/1e6:.0f} МПа'
                                    })
                                    
                                    print(f"    УСПЕХ ОДИН ПОРОХ!!!!!!!!: {powder}")
                                    print(f"   ω={omega_sum:.2f}кг, ω/q={omega_sum/q:.2f}")
                                    print(f"   Z_b1={Z_b1} | V={v_pm:.1f}м/с | Pmax={p_max/1e6:.1f}МПа")
                                    print(f"   V(-50°C)={v_pm_cold:.1f}м/с | P(+50°C)={p_mz_hot/1e6:.1f}МПа")

                        print(f" {powder} | ω={omega_sum:.2f}кг | V={v_pm:.1f}м/с | P={p_max/1e6:.1f}МПа | Z={Z_b1} |")
                            
                    except Exception as e:
                        print(f" Ошибка расчета для {powder}: {e}")
                    
                    pbar.update(1)
    
    print(f"\n{'='*60}")
    print(f"НАЙДЕНО УСПЕШНЫХ РЕШЕНИЙ (один порох): {len(successful_solutions)}")
    print(f"{'='*60}")
    
    if successful_solutions:
        successful_solutions.sort(key=lambda x: x['Z_b1'])
    
    return successful_solutions


def calculation_single_powder_lagrange(db, omega_sum_range, delta_range):
    """Расчет для одного пороха в ГД постановке"""
    
    initialize_result_file("result_single_lagrange.txt")

    total_iterations = len(db) * len(omega_sum_range) * len(delta_range)
    
    with tqdm(total=total_iterations, desc="Расчет одного пороха ГД") as pbar:

        for powder in db:
            powder_data = get_powder_data(powder)
            
            if powder_data is None:
                print(f"Пропускаем порох {powder} - нет данных")
                pbar.update(len(omega_sum_range) * len(delta_range))
                continue
                
            for omega_sum in omega_sum_range:
                if omega_sum / q < 0.1 or omega_sum / q > 2:
                    pbar.update(len(delta_range))
                    continue
                    
                for delta in delta_range:
                    W_0 = omega_sum / delta

                    if W_0 <= 0:
                        pbar.update(1)
                        continue

                    mass = omega_sum
                    
                    opts['powders'] = [
                        {'omega': mass, 'dbname': powder}
                    ]

                    f_sum = powder_data['f']
                    delta_sum = powder_data['delta']
                    b_sum = powder_data['b']
                    k_sum = powder_data['k']

                    opts['init_conditions']['W_0'] = W_0

                    try:
                        # Газодинамический расчет
                        result_lagrange = ozvb_lagrange(opts)
                        
                        p_max = max([max(layer['p']) for layer in result_lagrange['layers']])
                        last_layer = result_lagrange['layers'][-1]
                        v_muzzle = last_layer['u'][-1]
                        barrel_length = last_layer['x'][-1]
                        x_pm = result_lagrange['x_p'][-1] if 'x_p' in result_lagrange else barrel_length

                        x_e = x_e_func_lagrange(result_lagrange)
                        Z_b1, Pi, phi, E_m, eta_rm, eta_re, Lambda_e, zeta, Lamda_m, N_s, a, b, c = Z_b1_func_lagrange(
                            x_e, f_sum, omega_sum, delta, delta_sum, b_sum, k_sum, result_lagrange)
                        
                        base_conditions_met = (p_max <= 390000000.0 and v_muzzle >= 950 and x_pm >= 1.5 and x_pm < 5.625)
                        
                        if base_conditions_met:
                            # Проверка при -50°C
                            opts['stop_conditions']['T_0'] = 223.15
                            result_cold = ozvb_lagrange(opts)
                            last_layer_cold = result_cold['layers'][-1]
                            v_muzzle_cold = last_layer_cold['u'][-1]
                            
                            if v_muzzle_cold >= 800:
                                # Проверка при +50°C
                                opts['stop_conditions']['T_0'] = 323.15
                                result_hot = ozvb_lagrange(opts)
                                last_layer_hot = result_hot['layers'][-1]
                                p_mz_hot = last_layer_hot['p'][-1]
                                
                                if p_mz_hot <= 180000000:

                                    solution = {
                                        'powder1': powder,
                                        'powder2': '',
                                        'mass1': mass,
                                        'mass2': 0,
                                        'alpha': 1.0,
                                        'omega_sum': omega_sum,
                                        'omega_q_ratio': omega_sum / q,
                                        'delta': delta,
                                        'W_0': W_0,
                                        'Z_b1': Z_b1,
                                        'velocity': v_muzzle,
                                        'velocity_cold': v_muzzle_cold,
                                        'velocity_hot': result_hot['layers'][-1]['u'][-1],
                                        'pressure_max': p_max,
                                        'pressure_hot': p_mz_hot,
                                        'x_p': x_pm,
                                        'x_e': x_e,
                                        'barrel_length': barrel_length,
                                        'f_sum': f_sum,
                                        'delta_sum': delta_sum,
                                        'b_sum': b_sum,
                                        'k_sum': k_sum
                                    }
                                    successful_solutions.append(solution)
                                    
                                    save_solution_to_file(solution, "result_single_lagrange.txt")
                                    
                                    pbar.set_postfix({
                                        'Порох': f'{powder}',
                                        'Z_b1': f'{Z_b1}',
                                        'V': f'{v_muzzle:.0f} м/с',
                                        'V(-50)': f'{v_muzzle_cold:.0f} м/с',
                                        'P(+50)': f'{p_mz_hot/1e6:.0f} МПа'
                                    })
                                    
                                    print(f"    УСПЕХ ОДИН ПОРОХ ГД!!!!!!!!: {powder}")
                                    print(f"   ω={omega_sum:.2f}кг, ω/q={omega_sum/q:.2f}")
                                    print(f"   Z_b1={Z_b1} | V_дульн={v_muzzle:.1f}м/с | Pmax={p_max/1e6:.1f}МПа")
                                    print(f"   V(-50°C)={v_muzzle_cold:.1f}м/с | P(+50°C)={p_mz_hot/1e6:.1f}МПа")
                                    print(f"   Длина ствола: {barrel_length:.3f} м")

                        print(f" {powder} | ω={omega_sum:.2f}кг | V_дульн={v_muzzle:.1f}м/с | P={p_max/1e6:.1f}МПа | Z={Z_b1} |")
                            
                    except Exception as e:
                        print(f" Ошибка ГД расчета для {powder}: {e}")
                    
                    pbar.update(1)
    
    print(f"\n{'='*60}")
    print(f"НАЙДЕНО УСПЕШНЫХ РЕШЕНИЙ ГД (один порох): {len(successful_solutions)}")
    print(f"{'='*60}")
    
    if successful_solutions:
        successful_solutions.sort(key=lambda x: x['Z_b1'])
    
    return successful_solutions


# # Пример использования:
# # solutions_single = calculation_single_powder(db, omega_sum_range, delta_range)
# solutions_single_lagrange = calculation_single_powder_lagrange(db, omega_sum_range, delta_range)

solutions_lagrange = calculation_2_lagrange(db, omega_sum_range, delta_range, alpha_range)

# solutions = calculation_2(db, omega_sum_range, delta_range, alpha_range)