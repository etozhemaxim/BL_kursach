from pyballistics import ozvb_termo, get_options_sample
import numpy as np
from tqdm import tqdm

mass_distr = np.linspace(2,5,3)
vardelta = np.linspace(500,1200,5)


class Powders:

    def __init__(self):
        self.powders = ['ДГ-4 14/1', '4/1 фл', '5/1', 'ДГ-3 13/1', '4/1 фл', '5/1', 'ДГ-4 15/1', '4/1 фл', '5/1', 'АПЦ-235 П 16/1', '4/1 фл', '5/1', 'ДГ-3 14/1', '4/1 фл', '5/1', 'МАП-1 23/1', '4/1 фл', '5/1', 'БНГ-1355 25/1', '4/1 фл', '5/1', 'НДТ-3 16/1', '4/1 фл', '5/1', 'ДГ-2 15/1', '4/1 фл', '5/1', 'УГФ-1', '4/1 фл', '5/1', 'УГ-1', '4/1 фл', '5/1', 'ДГ-3 17/1', '4/1 фл', '5/1', 'НДТ-2 16/1', '4/1 фл', '5/1', 'НДТ-3 18/1', '4/1 фл', '5/1', 'ДГ-3 18/1', '4/1 фл', '5/1', 'ДГ-2 17/1', '4/1 фл', '5/1', 'НДТ-3 19/1', '4/1 фл', '5/1', 'ДГ-3 20/1', '4/1 фл', '5/1', 'НДТ-2 19/1', '4/1 фл', '5/1', '12/1 тр МН', '4/1 фл', '5/1', '7/1 УГ', '4/1 фл', '5/1', '15/1 тр В/А', '4/1 фл', '5/1', '8/1 УГ', '4/1 фл', '5/1', '16/1 тр В/А', '4/1 фл', '5/1', '11/1 БП', '4/1 фл', '5/1', '12/1 тр БП', '4/1 фл', '5/1', '18/1 тр', '4/1 фл', '5/1', '16/1 тр', '4/1 фл', '5/1', '22/1 тр', '4/1 фл', '5/1', '11/1 УГ', '4/1 фл', '5/1', '12/1 УГ', '4/1 фл', '5/1', '18/1 тр БП', '4/1 фл', '5/1', '9/7 МН', '4/1 фл', '5/1', '12/7', '4/1 фл', '5/1', '14/7 В/А', '4/1 фл', '5/1', '15/7', '4/1 фл', '5/1', '9/7 БП', '4/1 фл', '5/1', '14/7', '4/1 фл', '5/1', '17/7', '4/1 фл', '5/1', '14/7 БП', '4/1 фл', '5/1']
    
powders_DB = Powders()
db = powders_DB.powders[:5]


opts = get_options_sample()

def calculation_2 (db,mass_distr, vardelta ):

    for powder1 in db:
        for mass1 in mass_distr:
            for powder2 in db:
                for mass2 in mass_distr:
                    for delta in vardelta:

                        omega_sum = mass1 + mass2
                        W_0 = omega_sum / delta

                        if W_0 <= 0 or omega_sum <= 0:
                            print(f"Пропускаем: масса={omega_sum}, delta={delta}, W_0={W_0}")
                            continue  # переходим к следующей итераци
                        opts = get_options_sample()

                        opts['powders'] = [
                            {'omega': mass1 , 'dbname': powder1},
                            {'omega': mass2 , 'dbname': powder2}
                        ]
                        opts['init_conditions']['q'] = 5
                        opts['init_conditions']['d'] = 0.085
                        opts['init_conditions']['W_0'] = W_0
                        opts['init_conditions']['phi_1'] = 1.04
                        opts['init_conditions']['p_0'] = 30000000.0

                        opts['stop_conditions']['v_p'] = 900
                        opts['stop_conditions']['p_max'] = 390000000.0
                        opts['stop_conditions']['x_p'] = 5.625

                        result = ozvb_termo(opts)
                        # print(result)

                        x_pm = result['x_p'][-1]
                        vel = result['v_p'][-1]
                        pressure = max(result['p_m'])
                        print(vel, pressure, x_pm)


def Z_b1():

    
    return Z_b1


print(calculation_2(db,mass_distr,vardelta))
