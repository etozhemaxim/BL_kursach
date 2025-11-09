import matplotlib.pyplot as plt
import numpy as np
from pyballistics import get_options_sample, ozvb_lagrange
from matplotlib.lines import Line2D

# Получаем параметры и запускаем расчет
opts = get_options_sample()

q = opts['init_conditions']['q'] = 5
d = opts['init_conditions']['d'] = 0.085
phi_1 = opts['init_conditions']['phi_1'] = 1.04
p_0 = opts['init_conditions']['p_0'] = 30000000.0
opts['stop_conditions']['v_p'] = 950
opts['stop_conditions']['p_max'] = 390000000.0
opts['stop_conditions']['x_p'] = 3.628
opts['init_conditions']['W_0'] = 0.010524
opts['init_conditions']['T_0'] = 323.15 # +50
# opts['init_conditions']['T_0'] = 223.15 # -50
opts['powders'] = [
    {'omega': 3.3 , 'dbname': '7/1 УГ'},
    {'omega': 2.2 , 'dbname': '14/7 В/А'}
]

result = ozvb_lagrange(opts)

# Настройка стиля графиков
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 16
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = "Times New Roman"
plt.rcParams['mathtext.it'] = "Times New Roman:italic"

# ФЛАГИ ДЛЯ УПРАВЛЕНИЯ ЛИНИЯМИ - легко меняйте на True/False
SHOW_HORIZONTAL_LINES = True   # Отображать горизонтальные линии
SHOW_PRESSURE_LINE =True     # Линия давления 180 МПа
SHOW_VELOCITY_LINE = False     # Линия скорости 830 м/с
SHOW_PRESSURE_LINE_390 = False
SHOW_VELOCITY_LINE_950 = False
SHOW_LENGTH_LINE_5625 = False
# ЗНАЧЕНИЯ ЛИНИЙ - легко меняйте значения
PRESSURE_LINE_VALUE = 180     # МПа
VELOCITY_LINE_VALUE = 830     # м/с
SHOW_PRESSURE_VALUE_390 = 390000000
SHOW_VELOCITY_VALUE_950 = 950
SHOW_LENGTH_VALUE_5625 = 5.625
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

# Данные по горению порохов для разных позиций
# Дно снаряда (последний элемент массивов)
z_1_projectile = np.array([layer['z_1'][-1] for layer in result['layers']])
psi_1_projectile = np.array([layer['psi_1'][-1] for layer in result['layers']])
z_2_projectile = np.array([layer['z_2'][-1] for layer in result['layers']])
psi_2_projectile = np.array([layer['psi_2'][-1] for layer in result['layers']])

# Дно канала (первый элемент массивов)
z_1_breech = np.array([layer['z_1'][0] for layer in result['layers']])
psi_1_breech = np.array([layer['psi_1'][0] for layer in result['layers']])
z_2_breech = np.array([layer['z_2'][0] for layer in result['layers']])
psi_2_breech = np.array([layer['psi_2'][0] for layer in result['layers']])

# Находим особые точки для порохов - берем СРЕДНЕЕ время между порохами
# Дно снаряда - распад (берем z >= 1 для любого из порохов)
z_end_decay_projectile = np.where((z_1_projectile >= 1) | (z_2_projectile >= 1))[0]
z_end_decay_projectile_idx = z_end_decay_projectile[0] if len(z_end_decay_projectile) > 0 else len(times) - 1

# Дно снаряда - сгорание (берем psi >= 1 для любого из порохов)
psi_end_burn_projectile = np.where((psi_1_projectile >= 1) | (psi_2_projectile >= 1))[0]
psi_end_burn_projectile_idx = psi_end_burn_projectile[0] if len(psi_end_burn_projectile) > 0 else len(times) - 1

# Дно канала - распад (берем z >= 1 для любого из порохов)
z_end_decay_breech = np.where((z_1_breech >= 1) | (z_2_breech >= 1))[0]
z_end_decay_breech_idx = z_end_decay_breech[0] if len(z_end_decay_breech) > 0 else len(times) - 1

# Дно канала - сгорание (берем psi >= 1 для любого из порохов)
psi_end_burn_breech = np.where((psi_1_breech >= 1) | (psi_2_breech >= 1))[0]
psi_end_burn_breech_idx = psi_end_burn_breech[0] if len(psi_end_burn_breech) > 0 else len(times) - 1

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

# Строим графики
p1, = host.plot(times * 1e3, p_projectile / 1e6, color=color1, label='$p_p(t)$')
p2, = host.plot(times * 1e3, p_breech / 1e6, color=color2, linestyle='--', label='$p_b(t)$')
p3, = par1.plot(times * 1e3, v_projectile, color=color3, label='$\\upsilon_p(t)$')
p4, = par2.plot(times * 1e3, x_projectile, color=color4, label='$x_p(t)$')

# ДОБАВЛЯЕМ ГОРИЗОНТАЛЬНЫЕ ЛИНИИ (если включены флаги)
if SHOW_HORIZONTAL_LINES:
    if SHOW_PRESSURE_LINE:
        # Горизонтальная линия давления 180 МПа
        pressure_line = host.axhline(y=PRESSURE_LINE_VALUE, color='purple', linestyle='-', 
                                   alpha=0.7, linewidth=1.5, 
                                   label=f'$p$ = {PRESSURE_LINE_VALUE} МПа')
    
    if SHOW_VELOCITY_LINE:
        # Горизонтальная линия скорости 830 м/с
        velocity_line = par1.axhline(y=VELOCITY_LINE_VALUE, color='black', linestyle='-', 
                                   alpha=0.7, linewidth=1.5, 
                                   label=f'$\\upsilon_p$ = {VELOCITY_LINE_VALUE} м/с')
    if SHOW_PRESSURE_LINE_390:
        # Горизонтальная линия скорости 830 м/с
        pressure_line_390 = par1.axhline(y=SHOW_PRESSURE_VALUE_390, color='darkgreen', linestyle='-', 
                                   alpha=0.7, linewidth=1.5, 
                                   label=f'$\\upsilon_p$ = {SHOW_PRESSURE_VALUE_390} м/с')
    if SHOW_VELOCITY_LINE_950:
        # Горизонтальная линия скорости 830 м/с
        velocity_line_950 = par1.axhline(y=SHOW_VELOCITY_VALUE_950, color='#FF00FF', linestyle='-', 
                                   alpha=0.7, linewidth=1.5, 
                                   label=f'$\\upsilon_p$ = {SHOW_VELOCITY_VALUE_950} м/с')
    if SHOW_LENGTH_LINE_5625:
        # Горизонтальная линия скорости 830 м/с
        lenght_line_5625 = par1.axhline(y=SHOW_LENGTH_VALUE_5625, color='#A52A2A', linestyle='-', 
                                   alpha=0.7, linewidth=1.5, 
                                   label=f'$\\upsilon_p$ = {SHOW_LENGTH_VALUE_5625} м/с')

# # Добавляем только 4 маркера на график
# # Дно снаряда - распад (кружок)
# host.plot(times[z_end_decay_projectile_idx] * 1e3, p_projectile[z_end_decay_projectile_idx] / 1e6, 
#          'o', color='blue', markersize=8, markeredgewidth=1.5, markeredgecolor='black')

# # Дно снаряда - сгорание (квадрат)
# host.plot(times[psi_end_burn_projectile_idx] * 1e3, p_projectile[psi_end_burn_projectile_idx] / 1e6, 
#          's', color='green', markersize=8, markeredgewidth=1.5, markeredgecolor='black')

# # Дно канала - распад (треугольник вверх)
# host.plot(times[z_end_decay_breech_idx] * 1e3, p_breech[z_end_decay_breech_idx] / 1e6, 
#          '^', color='purple', markersize=8, markeredgewidth=1.5, markeredgecolor='black')

# # Дно канала - сгорание (треугольник вниз)
# host.plot(times[psi_end_burn_breech_idx] * 1e3, p_breech[psi_end_burn_breech_idx] / 1e6, 
#          'v', color='brown', markersize=8, markeredgewidth=1.5, markeredgecolor='black')

# Создаем кастомные элементы для легенды
legend_elements = [
    # Основные линии графиков
    p1, p2, p3, p4,
    #     # Маркеры для особых точек (только 4!)
    # Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, 
    #        markeredgewidth=1.5, markeredgecolor='black',
    #        label='распад пороха (дно снаряда)'),
    # Line2D([0], [0], marker='s', color='w', markerfacecolor='green', markersize=10,
    #        markeredgewidth=1.5, markeredgecolor='black',
    #        label='сгорание пороха (дно снаряда)'),
    # Line2D([0], [0], marker='^', color='w', markerfacecolor='purple', markersize=10,
    #        markeredgewidth=1.5, markeredgecolor='black',
    #        label='распад пороха (дно канала)'),
    # Line2D([0], [0], marker='v', color='w', markerfacecolor='brown', markersize=10,
    #        markeredgewidth=1.5, markeredgecolor='black',
    #        label='сгорание пороха (дно канала)')
]

# Добавляем горизонтальные линии в легенду (если они отображены)
if SHOW_HORIZONTAL_LINES:
    if SHOW_PRESSURE_LINE:
        legend_elements.append(pressure_line)
    if SHOW_VELOCITY_LINE:
        legend_elements.append(velocity_line)

# # Маркеры для особых точек (только 4!)
#     Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, 
#            markeredgewidth=1.5, markeredgecolor='black',
#            label='распад пороха (дно снаряда)'),
#     Line2D([0], [0], marker='s', color='w', markerfacecolor='green', markersize=10,
#            markeredgewidth=1.5, markeredgecolor='black',
#            label='сгорание пороха (дно снаряда)'),
#     Line2D([0], [0], marker='^', color='w', markerfacecolor='purple', markersize=10,
#            markeredgewidth=1.5, markeredgecolor='black',
#            label='распад пороха (дно канала)'),
#     Line2D([0], [0], marker='v', color='w', markerfacecolor='brown', markersize=10,
#            markeredgewidth=1.5, markeredgecolor='black',
#            label='сгорание пороха (дно канала)')

# Создаем легенду со всеми элементами
host.legend(handles=legend_elements, loc='upper left')
# Опустить легенду на 20% от верха  
host.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 0.8))
# Сдвигаем правую ось для третьего графика
par2.spines['right'].set_position(('outward', 60))

# Устанавливаем цвета подписей осей
host.yaxis.label.set_color('black')
par1.yaxis.label.set_color(color3)
par2.yaxis.label.set_color(color4)

host.grid()
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
print(f"  Распад пороха (дно снаряда): {times[z_end_decay_projectile_idx]*1000:.3f} мс")
print(f"  Сгорание пороха (дно снаряда): {times[psi_end_burn_projectile_idx]*1000:.3f} мс")
print(f"  Распад пороха (дно канала): {times[z_end_decay_breech_idx]*1000:.3f} мс")
print(f"  Сгорание пороха (дно канала): {times[psi_end_burn_breech_idx]*1000:.3f} мс")