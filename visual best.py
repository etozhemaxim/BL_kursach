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

# Устанавливаем пределы осей
max_time = max(times) * 1000
host.set_xlim(0, max_time * 1.05)

plim = (0, 300)
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

plt.title('Газодинамическое моделирование: ДГ-2 17/1 + 12/1 УГ')
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