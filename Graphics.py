import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

# Загрузка данных
df = pd.read_csv('result_lagrange.txt')

# Проверка загрузки данных
print("Первые 5 строк данных:")
print(df.head())
print("\nИнформация о данных:")
print(df.info())

# Сортировка по критерию Слухоцкого (лучшие решения с большими значениями Z_b1)
df_sorted = df.sort_values('Z_b1', ascending=False)
best_solution = df_sorted.iloc[0]  # Лучшее решение

print(f"\nЛучшее решение по критерию Слухоцкого:")
print(f"Z_b1 = {best_solution['Z_b1']:.2e}")
print(f"Давление = {best_solution['pressure_max_mpa']:.2f} МПа")
print(f"Длина ствола = {best_solution['x_p_m']:.3f} м")
print(f"Плотность заряжания = {best_solution['delta_kg_m3']:.1f} кг/м³")
print(f"Omega_q_ratio = {best_solution['omega_q_ratio']:.3f}")
print(f"Смесь: {best_solution['mixture_name']}")

# ГРАФИК 1: Зависимость давления от длины ствола
plt.figure(figsize=(14, 8))

scatter1 = plt.scatter(df['x_p_m'], 
                      df['pressure_max_mpa'], 
                      c=df['Z_b1'], 
                      cmap='plasma', 
                      norm=LogNorm(vmin=df['Z_b1'].min(), vmax=df['Z_b1'].max()),
                      alpha=0.8,
                      s=60,
                      edgecolors='white',
                      linewidth=0.3)

# Пометка лучшего решения на первом графике
plt.scatter(best_solution['x_p_m'], 
           best_solution['pressure_max_mpa'], 
           color='red', 
           marker='*', 
           s=400,
           edgecolors='black',
           linewidth=1.5,
           label=f'Лучшее решение ')

# # Аннотация для лучшего решения на первом графике
# plt.annotate(f'Лучшее: Z_b1 = {best_solution["Z_b1"]:.2e}\n'
#             f'P = {best_solution["pressure_max_mpa"]:.1f} МПа\n'
#             f'L = {best_solution["x_p_m"]:.2f} м',
#             xy=(best_solution['x_p_m'], best_solution['pressure_max_mpa']),
#             xytext=(15, 15),
#             textcoords='offset points',
#             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
#             fontsize=10,
#             fontweight='bold')

# Настройки первого графика
plt.xlabel('Длина ствола,м', fontsize=14, fontweight='bold')
plt.ylabel('Максимальное давление, МПа', fontsize=14, fontweight='bold')
plt.title('', 
         fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(loc='upper left', fontsize=12)

# Цветовая шкала для первого графика
cbar1 = plt.colorbar(scatter1, location='right', shrink=0.8)
cbar1.set_label('Критерий Слухоцкого', fontsize=12, fontweight='bold')

# Улучшение внешнего вида
plt.gca().set_facecolor('#f8f9fa')
plt.tight_layout()

# Показать первый график
plt.show()

# ГРАФИК 2: Зависимость плотности заряжания от omega_q_ratio
plt.figure(figsize=(14, 8))

scatter2 = plt.scatter(df['omega_q_ratio'], 
                      df['delta_kg_m3'], 
                      c=df['Z_b1'], 
                      cmap='plasma', 
                      norm=LogNorm(vmin=df['Z_b1'].min(), vmax=df['Z_b1'].max()),
                      alpha=0.8,
                      s=60,
                      edgecolors='white',
                      linewidth=0.3)

# Пометка лучшего решения на втором графике
plt.scatter(best_solution['omega_q_ratio'], 
           best_solution['delta_kg_m3'], 
           color='red', 
           marker='*', 
           s=400,
           edgecolors='black',
           linewidth=1.5,
           label=f'Лучшее решение ')

# # Аннотация для лучшего решения на втором графике
# plt.annotate(f'Лучшее: Z_b1 = {best_solution["Z_b1"]:.2e}\n'
#             f'δ = {best_solution["delta_kg_m3"]:.1f} кг/м³\n'
#             f'ω_q = {best_solution["omega_q_ratio"]:.3f}',
#             xy=(best_solution['omega_q_ratio'], best_solution['delta_kg_m3']),
#             xytext=(15, 15),
#             textcoords='offset points',
#             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
#             fontsize=10,
#             fontweight='bold')

plt.rcParams['text.usetex'] = False  # Обычно лучше оставить False
plt.rcParams['mathtext.fontset'] = 'stix'  # или 'cm', 'dejavusans'

# Настройки второго графика
plt.xlabel(r'$\omega / q$', fontsize=14, fontweight='bold')
plt.ylabel(r'$\Delta$, кг/м$^3$', fontsize=14, fontweight='bold')
plt.title('', 
         fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(loc='upper left', fontsize=12)

# Цветовая шкала для второго графика
cbar2 = plt.colorbar(scatter2, location='right', shrink=0.8)
cbar2.set_label('Критерий Слухоцкого ', fontsize=12, fontweight='bold')

# Улучшение внешнего вида
plt.gca().set_facecolor('#f8f9fa')
plt.tight_layout()

# Показать второй график
plt.show()

# # ГРАФИК 3: Зависимость критерия Слухоцкого от длины ствола с цветом по давлению
# plt.figure(figsize=(14, 8))

# scatter3 = plt.scatter(df['x_p_m'], 
#                       df['Z_b1'], 
#                       c=df['pressure_max_mpa'], 
#                       cmap='plasma', 
#                       norm=LogNorm(vmin=df['pressure_max_mpa'].min(), vmax=df['pressure_max_mpa'].max()),
#                       alpha=0.8,
#                       s=60,
#                       edgecolors='white',
#                       linewidth=0.3)

# # Пометка лучшего решения на третьем графике
# plt.scatter(best_solution['x_p_m'], 
#            best_solution['Z_b1'], 
#            color='red', 
#            marker='*', 
#            s=400,
#            edgecolors='black',
#            linewidth=1.5,
#            label=f'Лучшее решение ')

# # Настройки третьего графика
# plt.xlabel('Длина ствола, м', fontsize=14, fontweight='bold')
# plt.ylabel('Критерий Слухоцкого', fontsize=14, fontweight='bold')
# plt.title('', 
#          fontsize=16, fontweight='bold')
# plt.grid(True, alpha=0.3, linestyle='--')
# plt.legend(loc='upper left', fontsize=12)

# # Цветовая шкала для третьего графика
# cbar3 = plt.colorbar(scatter3, location='right', shrink=0.8)
# cbar3.set_label('Максимальное давление, МПа', fontsize=12, fontweight='bold')

# # Улучшение внешнего вида
# plt.gca().set_facecolor('#f8f9fa')
# plt.tight_layout()

# # Показать третий график
# plt.show()


# Дополнительная статистика
print(f"\nСтатистика по данным:")
print(f"Количество точек: {len(df)}")
print(f"Диапазон длины ствола: {df['x_p_m'].min():.3f} - {df['x_p_m'].max():.3f} м")
print(f"Диапазон давления: {df['pressure_max_mpa'].min():.2f} - {df['pressure_max_mpa'].max():.2f} МПа")
print(f"Диапазон плотности заряжания: {df['delta_kg_m3'].min():.1f} - {df['delta_kg_m3'].max():.1f} кг/м³")
print(f"Диапазон omega_q_ratio: {df['omega_q_ratio'].min():.3f} - {df['omega_q_ratio'].max():.3f}")
print(f"Диапазон Z_b1: {df['Z_b1'].min():.2e} - {df['Z_b1'].max():.2e}")
print(f"Медиана Z_b1: {df['Z_b1'].median():.2e}")
print(f"Среднее Z_b1: {df['Z_b1'].mean():.2e}")

# Топ-5 лучших решений
print(f"\nТоп-5 лучших решений по критерию Слухоцкого:")
top_5 = df_sorted.head(10)
for i, (idx, row) in enumerate(top_5.iterrows(), 1):
    print(f"{i}. Z_b1 = {row['Z_b1']:.2e}, "
          f"Давление = {row['pressure_max_mpa']:.2f} МПа, "
          f"Длина = {row['x_p_m']:.3f} м, "
          f"Плотность = {row['delta_kg_m3']:.1f} кг/м³, "
          f"ω_q = {row['omega_q_ratio']:.3f}, "
          f"Смесь: {row['mixture_name']}")