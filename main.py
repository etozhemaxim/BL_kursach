from pyballistics import ozvb_termo, get_options_agard
import matplotlib.pyplot as plt
import numpy as np 

opts = get_options_agard() # получить словарь с начальными данными задачи AGARD
result = ozvb_termo(opts)  # произвести расчет и получить результат

 # если нет библиотеки matplotlib, то установить ее можно при помощи команды pip install matplotlib

plt.plot(result['t'], result['p_m']) # среднебаллистическое давление от времени
plt.grid()  # сетка на графике
plt.show()  # показать график
print(np.max(result['p_m']))



print(result['v_p'][-1])


# доля сгоревшего пороха
print(result['psi_1'][-1])
