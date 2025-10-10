from pyballistics import ozvb_lagrange, get_options_sample
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd

opts = get_options_sample() # получить словарь с начальными данными задачи AGARD
result = ozvb_lagrange(opts)  # произвести расчет и получить результат

# Преобразование словаря в DataFrame
df_opts = pd.DataFrame(list(opts.items()), columns=['Parameter', 'Value'])
print(df_opts)