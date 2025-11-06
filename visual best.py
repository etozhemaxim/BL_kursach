import matplotlib.pyplot as plt
import numpy as np
from pyballistics import get_options_sample, ozvb_lagrange

# Получаем параметры и запускаем расчет
opts = get_options_sample()

q = opts['init_conditions']['q'] = 5
d = opts['init_conditions']['d'] = 0.085
phi_1 = opts['init_conditions']['phi_1'] = 1.04
p_0 = opts['init_conditions']['p_0'] = 30000000.0
opts['stop_conditions']['v_p'] = 950
opts['stop_conditions']['p_max'] = 390000000.0
opts['stop_conditions']['x_p'] = 5.625
opts['init_conditions']['T_0'] = 323.15

opts['init_conditions']['W_0'] = 0.005016

opts['powders'] = [
    {'omega': 2.735 , 'dbname': 'ДГ-2 17/1'},
    {'omega': 2.122 , 'dbname': '12/1 УГ'}
                 ]

result = ozvb_lagrange(opts)

# Настройка стиля графиков
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 16
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = "Times New Roman"
plt.rcParams['mathtext.it'] = "Times New Roman:italic"

# Создаем фигуру и оси
fig = plt.figure(figsize=(12, 8))
host = fig.add_subplot(111)

# Создаем дополнительные оси
par1 = host.twinx()
par2 = host.twinx()

# Извлекаем данные из результатов
times = np.array([layer['t'] for layer in result['layers']])

# Давление на дне снаряда (последний элемент массива давления)
p_projectile = np.array([layer['p'][-1] for layer in result['layers']])

# Давление в казеннике (первый элемент массива давления)
p_breech = np.array([layer['p'][0] for layer in result['layers']])

# Скорость снаряда (последний элемент массива скорости)
v_projectile = np.array([layer['u'][-1] for layer in result['layers']])

# Положение снаряда (последний элемент массива координат)
x_projectile = np.array([layer['x'][-1] for layer in result['layers']])

# # Данные по горению порохов для разных позиций
# # Дно снаряда (последний элемент массивов)
# z_1_projectile = np.array([layer['z_1'][-1] for layer in result['layers']])  # ДГ-2 17/1
# psi_1_projectile = np.array([layer['psi_1'][-1] for layer in result['layers']])
# z_2_projectile = np.array([layer['z_2'][-1] for layer in result['layers']])  # 12/1 УГ
# psi_2_projectile = np.array([layer['psi_2'][-1] for layer in result['layers']])

# # Дно канала (первый элемент массивов)
# z_1_breech = np.array([layer['z_1'][0] for layer in result['layers']])  # ДГ-2 17/1
# psi_1_breech = np.array([layer['psi_1'][0] for layer in result['layers']])
# z_2_breech = np.array([layer['z_2'][0] for layer in result['layers']])  # 12/1 УГ
# psi_2_breech = np.array([layer['psi_2'][0] for layer in result['layers']])

# # Находим особые точки для порохов
# # Дно снаряда
# z1_end_decay_projectile = np.where(z_1_projectile >= 1)[0]
# z1_end_decay_projectile_idx = z1_end_decay_projectile[0] if len(z1_end_decay_projectile) > 0 else len(times) - 1

# z2_end_decay_projectile = np.where(z_2_projectile >= 1)[0]
# z2_end_decay_projectile_idx = z2_end_decay_projectile[0] if len(z2_end_decay_projectile) > 0 else len(times) - 1

# psi1_end_burn_projectile = np.where(psi_1_projectile >= 1)[0]
# psi1_end_burn_projectile_idx = psi1_end_burn_projectile[0] if len(psi1_end_burn_projectile) > 0 else len(times) - 1

# psi2_end_burn_projectile = np.where(psi_2_projectile >= 1)[0]
# psi2_end_burn_projectile_idx = psi2_end_burn_projectile[0] if len(psi2_end_burn_projectile) > 0 else len(times) - 1

# # Дно канала
# z1_end_decay_breech = np.where(z_1_breech >= 1)[0]
# z1_end_decay_breech_idx = z1_end_decay_breech[0] if len(z1_end_decay_breech) > 0 else len(times) - 1

# z2_end_decay_breech = np.where(z_2_breech >= 1)[0]
# z2_end_decay_breech_idx = z2_end_decay_breech[0] if len(z2_end_decay_breech) > 0 else len(times) - 1

# psi1_end_burn_breech = np.where(psi_1_breech >= 1)[0]
# psi1_end_burn_breech_idx = psi1_end_burn_breech[0] if len(psi1_end_burn_breech) > 0 else len(times) - 1

# psi2_end_burn_breech = np.where(psi_2_breech >= 1)[0]
# psi2_end_burn_breech_idx = psi2_end_burn_breech[0] if len(psi2_end_burn_breech) > 0 else len(times) - 1

# # Собираем все маркеры
# markers = [
#     z1_end_decay_projectile_idx, z2_end_decay_projectile_idx, 
#     psi1_end_burn_projectile_idx, psi2_end_burn_projectile_idx,
#     z1_end_decay_breech_idx, z2_end_decay_breech_idx,
#     psi1_end_burn_breech_idx, psi2_end_burn_breech_idx
# ]

# Устанавливаем пределы осей
max_time = max(times) * 1000
host.set_xlim(0, max_time * 1.05)

plim = (0, 400)
vlim = (0, 1000)
xlim = (0, 10)

host.set_ylim(*plim)
par1.set_ylim(*vlim)
par2.set_ylim(*xlim)

# Настраиваем засечки
n = 11
host.yaxis.set_ticks(np.linspace(*plim, n))
par1.yaxis.set_ticks(np.linspace(*vlim, n))
par2.yaxis.set_ticks(np.linspace(*xlim, n))

# Подписи осей
host.set_xlabel("Время, мс")
host.set_ylabel("$p$, МПа")
par1.set_ylabel("$\\upsilon_p$, м/с")
par2.set_ylabel("$x_p$, м")

# Цвета графиков
color1 = 'darkorange'  # давление на снаряде
color2 = 'red'         # давление в казеннике  
color3 = 'darkgreen'   # скорость
color4 = 'blue'        # положение

# Строим графики с маркерами
p1, = host.plot(times * 1e3, p_projectile / 1e6, color=color1, label='$p_p(t)$')
p2, = host.plot(times * 1e3, p_breech / 1e6, color=color2, linestyle='--', label='$p_b(t)$')
p3, = par1.plot(times * 1e3, v_projectile, color=color3, label='$\\upsilon_p(t)$')
p4, = par2.plot(times * 1e3, x_projectile, color=color4, label='$x_p(t)$')

# # Добавляем маркеры для особых точек (разными маркерами для дна снаряда и дна канала)
# # Дно снаряда - кружки
# host.plot(times[z1_end_decay_projectile_idx] * 1e3, p_projectile[z1_end_decay_projectile_idx] / 1e6, 
#          'o', color='blue', markersize=8, label='распад пороха (дно снаряда)')
# host.plot(times[z2_end_decay_projectile_idx] * 1e3, p_projectile[z2_end_decay_projectile_idx] / 1e6, 
#          'o', color='blue', markersize=8)

# host.plot(times[psi1_end_burn_projectile_idx] * 1e3, p_projectile[psi1_end_burn_projectile_idx] / 1e6, 
#          's', color='green', markersize=8, label='сгорание пороха (дно снаряда)')
# host.plot(times[psi2_end_burn_projectile_idx] * 1e3, p_projectile[psi2_end_burn_projectile_idx] / 1e6, 
#          's', color='green', markersize=8)

# # Дно канала - треугольники
# host.plot(times[z1_end_decay_breech_idx] * 1e3, p_breech[z1_end_decay_breech_idx] / 1e6, 
#          '^', color='purple', markersize=8, label='распад пороха (дно канала)')
# host.plot(times[z2_end_decay_breech_idx] * 1e3, p_breech[z2_end_decay_breech_idx] / 1e6, 
#          '^', color='purple', markersize=8)

# host.plot(times[psi1_end_burn_breech_idx] * 1e3, p_breech[psi1_end_burn_breech_idx] / 1e6, 
#          'v', color='brown', markersize=8, label='сгорание пороха (дно канала)')
# host.plot(times[psi2_end_burn_breech_idx] * 1e3, p_breech[psi2_end_burn_breech_idx] / 1e6, 
#          'v', color='brown', markersize=8)

# Объединяем все графики в одну легенду
lns = [p1, p2, p3, p4]
host.legend(handles=lns, loc='upper left')

# Сдвигаем правую ось для третьего графика
par2.spines['right'].set_position(('outward', 60))

# Устанавливаем цвета подписей осей
host.yaxis.label.set_color('black')
par1.yaxis.label.set_color(color3)
par2.yaxis.label.set_color(color4)

host.grid()

# plt.title('Газодинамическое моделирование: ДГ-2 17/1 + 12/1 УГ')
plt.tight_layout()
plt.show()

# Выводим дополнительную информацию
print(f"Причина остановки: {result['stop_reason']}")
print(f"Время расчета: {result['execution_time']:.3f} с")
print(f"Максимальное давление на снаряде: {max(p_projectile)/1e6:.1f} МПа")
print(f"Максимальное давление в казеннике: {max(p_breech)/1e6:.1f} МПа")
print(f"Максимальная скорость: {max(v_projectile):.1f} м/с")
print(f"Конечное положение снаряда: {x_projectile[-1]:.3f} м")
print(f"Количество шагов расчета: {len(result['layers'])}")
print("\nОсобые точки горения порохов:")
print("Дно снаряда:")
# print(f"  Распад ДГ-2 17/1: {times[z1_end_decay_projectile_idx]*1000:.3f} мс")
# print(f"  Распад 12/1 УГ: {times[z2_end_decay_projectile_idx]*1000:.3f} мс")
# print(f"  Сгорание ДГ-2 17/1: {times[psi1_end_burn_projectile_idx]*1000:.3f} мс")
# print(f"  Сгорание 12/1 УГ: {times[psi2_end_burn_projectile_idx]*1000:.3f} мс")
# print("Дно канала:")
# print(f"  Распад ДГ-2 17/1: {times[z1_end_decay_breech_idx]*1000:.3f} мс")
# print(f"  Распад 12/1 УГ: {times[z2_end_decay_breech_idx]*1000:.3f} мс")
# print(f"  Сгорание ДГ-2 17/1: {times[psi1_end_burn_breech_idx]*1000:.3f} мс")
# print(f"  Сгорание 12/1 УГ: {times[psi2_end_burn_breech_idx]*1000:.3f} мс")