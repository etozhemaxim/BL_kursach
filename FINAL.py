from pyballistics import ozvb_termo, get_options_sample
import numpy as np
from tqdm import tqdm

from baza_powder import powders_db

mass_distr = np.linspace(2,5,3)
vardelta_db = np.linspace(500,1200,5)


class Powders:

    def __init__(self):
        self.powders = ['ДГ-4 14/1', '4/1 фл', '5/1', 'ДГ-3 13/1', '4/1 фл', '5/1', 'ДГ-4 15/1', '4/1 фл', '5/1', 'АПЦ-235 П 16/1', '4/1 фл', '5/1', 'ДГ-3 14/1', '4/1 фл', '5/1', 'МАП-1 23/1', '4/1 фл', '5/1', 'БНГ-1355 25/1', '4/1 фл', '5/1', 'НДТ-3 16/1', '4/1 фл', '5/1', 'ДГ-2 15/1', '4/1 фл', '5/1', 'УГФ-1', '4/1 фл', '5/1', 'УГ-1', '4/1 фл', '5/1', 'ДГ-3 17/1', '4/1 фл', '5/1', 'НДТ-2 16/1', '4/1 фл', '5/1', 'НДТ-3 18/1', '4/1 фл', '5/1', 'ДГ-3 18/1', '4/1 фл', '5/1', 'ДГ-2 17/1', '4/1 фл', '5/1', 'НДТ-3 19/1', '4/1 фл', '5/1', 'ДГ-3 20/1', '4/1 фл', '5/1', 'НДТ-2 19/1', '4/1 фл', '5/1', '12/1 тр МН', '4/1 фл', '5/1', '7/1 УГ', '4/1 фл', '5/1', '15/1 тр В/А', '4/1 фл', '5/1', '8/1 УГ', '4/1 фл', '5/1', '16/1 тр В/А', '4/1 фл', '5/1', '11/1 БП', '4/1 фл', '5/1', '12/1 тр БП', '4/1 фл', '5/1', '18/1 тр', '4/1 фл', '5/1', '16/1 тр', '4/1 фл', '5/1', '22/1 тр', '4/1 фл', '5/1', '11/1 УГ', '4/1 фл', '5/1', '12/1 УГ', '4/1 фл', '5/1', '18/1 тр БП', '4/1 фл', '5/1', '9/7 МН', '4/1 фл', '5/1', '12/7', '4/1 фл', '5/1', '14/7 В/А', '4/1 фл', '5/1', '15/7', '4/1 фл', '5/1', '9/7 БП', '4/1 фл', '5/1', '14/7', '4/1 фл', '5/1', '17/7', '4/1 фл', '5/1', '14/7 БП', '4/1 фл', '5/1']
    
powders_DB = Powders()
available_powders = [p for p in powders_DB.powders[:5] if p in powders_db]
db = available_powders


opts = get_options_sample()
q = opts['init_conditions']['q'] = 5
d = opts['init_conditions']['d'] = 0.085
phi_1 = opts['init_conditions']['phi_1'] = 1.04
p_0 = opts['init_conditions']['p_0'] = 30000000.0
opts['stop_conditions']['v_p'] = 1500
opts['stop_conditions']['p_max'] = 390000000.0
opts['stop_conditions']['x_p'] = 5.625


def calculation_2 (db,mass_distr, vardelta_db ):
    
    # Считаем общее количество итераций для прогресс-бара
    total_iterations = len(db) * len(mass_distr) * len(db) * len(mass_distr) * len(vardelta_db)
    
    # Создаем прогресс-бар
    with tqdm(total=total_iterations, desc="Расчет смесей") as pbar:

        for i, powder1 in enumerate(db):
            for mass1 in mass_distr:
                for j, powder2 in enumerate(db):
                    if i < j: 
                        for mass2 in mass_distr:
                            for vardelta in vardelta_db:
                                    omega_sum = mass1 + mass2
                                    W_0 = omega_sum / vardelta


                                    if W_0 <= 0 or omega_sum <= 0:
                                        print(f"Пропускаем: масса={omega_sum}, delta={vardelta}, W_0={W_0}")
                                        pbar.update(1)
                                        continue

                                    opts['powders'] = [
                                        {'omega': mass1 , 'dbname': powder1},
                                        {'omega': mass2 , 'dbname': powder2}
                                    ]

                                    powder1_data = powders_db[powder1]
                                    powder2_data = powders_db[powder2]

                                    fraction1 = mass1 / omega_sum
                                    fraction2 = mass2 / omega_sum

                                    f_sum = fraction1 * powder1_data['f'] + fraction2 * powder2_data['f']

                                    delta_sum = fraction1 *  powder1_data['delta'] + fraction2 * powder2_data['delta']

                                    b_sum = fraction1 *  powder1_data['b'] + fraction2 * powder2_data['b']

                                    k_sum= fraction1 *  powder1_data['k'] + fraction2 * powder2_data['k']                     

                                    opts['init_conditions']['W_0'] = W_0


                                    result1 = ozvb_termo(opts)
                                    # print(result)

                                    x_pm = result1['x_p'][-1]
                                    v_pm = result1['v_p'][-1]
                                    p_max = max(result1['p_m'])

                                    if (p_max <= 390000000.0 and v_pm >= 950 and x_pm >= 1.5 and x_pm < 5.525):

                                        opts['stop_conditions']['T_0'] = 223.15 #-50 градусов

                                        result = ozvb_termo(opts)

                                        v_pm = result['v_p'][-1]

                                        if v_pm  >= 830: 

                                            opts['stop_conditions']['T_0'] = 323.15 #50 градусов

                                            result = ozvb_termo(opts)

                                            p_mz = result['p_m'][-1]
                                            
                                            if  p_mz >= 180: 
                                                print(f"ты пидор")
                                                x_e = x_e_func(result1)
                                                Z_b1 = Z_b1_func(x_e, f_sum, omega_sum, vardelta, delta_sum, b_sum, k_sum, result1)
                                                print(Z_b1)
                                                print(f" {powder1} ({fraction1}) + {powder2}({fraction2}) {omega_sum} ==={Z_b1} ")


                                    
                                    pbar.update(1)


def Z_b1_func(x_e, f_sum, omega_sum, vardelta, delta, b, k, result):

    phi_1 = 1.04
    p_ign = 5e6
    omega_ign = 0.01
    K = 0.5
    v_p = result['v_p'][-1]
    x_p = result['x_p'][-1]

    Pi =  f_sum / (k - 1) 

    phi = phi_1 *  ( 1 / 3*q ) * (omega_ign * omega_sum)   

    E_m = ( q * v_p**2 ) / 2

    eta_rm = phi * E_m / Pi    

    eta_re = E_m / omega_sum

    Lambda_e = x_e / x_p

    zeta = ( p_ign / f_sum ) * ( ( 1 / vardelta ) - ( 1  / delta ) ) * ( 1 / 1 + ( (b * p_ign) / f_sum ) )

    Lamda_m = (1 - b * vardelta * (1 + zeta) + Lambda_e) * ( 1 + zeta - eta_re/ 1 + zeta - eta_rm)**(1 / k - 1) - (1 - b * vardelta * (1 + zeta))

    N_s = K * (1 + Lamda_m) * ( q / omega_sum )

    Z_b1 = ( q * v_p**2 / 2 )**5 * ( np.sqrt(N_s) / omega_sum * x_p)

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


print(calculation_2(db,mass_distr,vardelta_db))