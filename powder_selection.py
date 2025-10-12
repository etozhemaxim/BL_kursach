import math
from math import *
import itertools
import numpy as np
import matplotlib.pyplot as plt
# Константы
f = 0.968e6  # Дж/кг
k = 1.25
b = 1.113e-3  # м³/кг
# Генерация диапазонов параметров
vardelta_values = list(range(550, 700, 10))  # кг/м³, от 650 до 780 с шагом 10
eta_rm_values = list(np.arange(0.1, 0.6, 0.1))  # r_m от 0.1 до 0.5 с шагом 0.1
B_values = [4.5, 2.0]  # B_1 (трубчатый) и B_7 (семиканальный)
v_pm = 950  # дульная скорость
d = 0.085
# Параметры формы порохового зерна
z_e = 1
kappa = 1
lambda_val = 0
p_ign = 1e6
delta = 1520
q = 5  # масса снаряда, кг 
phi_1 = 1.02  # коэффициент фиктивности 
omega_ign = 0.01  # масса воспламенителя, кг
S = pi * d**2 / 4

def I_e(vardelta, r_m, B):
    """Расчет массы пороха и импульса"""
    zeta = (p_ign / f) * (1 / vardelta - 1 / delta) * 1 / (1 + (b * p_ign) / f)  
    
    omega = (phi_1 * q) / (((2 * f) / ((k - 1) * v_pm**2)) * r_m - (zeta + 1) / 3)
    
    phi = phi_1 * (1 / (3 * q)) * (omega_ign + omega)
    
    I_e = (math.sqrt(f * omega * phi * q * B) / S) / 10**6
    
    return omega, I_e

def vary_parameters():
    """Автоматическое варьирование параметров"""
    results = []
    
    # Создаем все комбинации параметров
    combinations = list(itertools.product(vardelta_values, eta_rm_values, B_values))
    
    print("Результаты автоматического варьирования:")
    print("=" * 85)
    print(f"{'vardelta':<10} {'r_m':<8} {'B':<8} {'omega (кг)':<15} {'I_e (МПа·с)':<15}")
    print("-" * 85)
    
    for vardelta, r_m, B in combinations:
        omega, impulse = I_e(vardelta, r_m, B)
        results.append({
            'vardelta': vardelta,
            'r_m': r_m,
            'B': B,
            'omega': omega,
            'I_e': impulse
        })
        
        print(f"{vardelta:<10} {r_m:<8.1f} {B:<8} {omega:<15.4f} {impulse:<15.4f}")
    
    print("=" * 85)
    print(f"Всего рассчитано комбинаций: {len(results)}")
    
    return results

def vary_parameters():
    """Автоматическое варьирование параметров"""
    results = []
    
    # Создаем все комбинации параметров
    combinations = list(itertools.product(vardelta_values, eta_rm_values, B_values))
    
    for vardelta, r_m, B in combinations:
        omega, impulse = I_e(vardelta, r_m, B)
        results.append({
            'vardelta': vardelta,
            'r_m': r_m,
            'B': B,
            'omega': omega,
            'I_e': impulse
        })
    
    # СОРТИРОВКА результатов по I_e (по возрастанию)
    results_sorted = sorted(results, key=lambda x: x['I_e'])
    
    print("Результаты автоматического варьирования (отсортированные по I_e):")
    print("=" * 85)
    print(f"{'vardelta':<10} {'r_m':<8} {'B':<8} {'omega (кг)':<15} {'I_e (МПа·с)':<15}")
    print("-" * 85)
    
    for result in results_sorted:
        print(f"{result['vardelta']:<10} {result['r_m']:<8.1f} {result['B']:<8} {result['omega']:<15.4f} {result['I_e']:<15.4f}")
    
    print("=" * 85)
    print(f"Всего рассчитано комбинаций: {len(results_sorted)}")
    
    return results_sorted

# Выполняем варьирование с сортировкой
results_sorted = vary_parameters()

# Создаем график I_e от omega (точки будут отсортированы)
plt.figure(figsize=(12, 8))

# Разделяем точки по значениям B для разных цветов
B_45_points = [r for r in results_sorted if r['B'] == 4.5]
B_20_points = [r for r in results_sorted if r['B'] == 2.0]

# Извлекаем данные для графиков
omega_45 = [r['omega'] for r in B_45_points]
I_e_45 = [r['I_e'] for r in B_45_points]

omega_20 = [r['omega'] for r in B_20_points]
I_e_20 = [r['I_e'] for r in B_20_points]

# Строим графики
plt.scatter(omega_45, I_e_45, c='red', alpha=0.7, s=12, label='B = 4.5 (трубчатый)')
plt.scatter(omega_20, I_e_20, c='blue', alpha=0.7, s=12, label='B = 2.0 (семиканальный)')

# Настройки графика
plt.xlabel('Масса пороха ω (кг)', fontsize=12)
plt.ylabel('Импульс I_e (МПа·с)', fontsize=12)
plt.title('Зависимость импульса I_e от массы пороха ω\n(Все точки варьирования параметров, отсортированные по I_e)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)

# Добавляем информацию о количестве точек
plt.text(0.02, 0.98, f'Всего точек: {len(results_sorted)}', 
         transform=plt.gca().transAxes, fontsize=10, 
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.text(0.02, 0.92, f'B=4.5: {len(B_45_points)} точек', 
         transform=plt.gca().transAxes, fontsize=10, 
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

plt.text(0.02, 0.86, f'B=2.0: {len(B_20_points)} точек', 
         transform=plt.gca().transAxes, fontsize=10, 
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout()
plt.show()

# Дополнительная статистика по отсортированным данным
print("\nСтатистика по отсортированным результатам:")
print(f"Всего точек: {len(results_sorted)}")
print(f"Точек с B=4.5: {len(B_45_points)}")
print(f"Точек с B=2.0: {len(B_20_points)}")
print(f"Диапазон ω: {min([r['omega'] for r in results_sorted]):.4f} - {max([r['omega'] for r in results_sorted]):.4f} кг")
print(f"Диапазон I_e: {min([r['I_e'] for r in results_sorted]):.4f} - {max([r['I_e'] for r in results_sorted]):.4f} МПа·с")

# Вывод первых 5 и последних 5 результатов для демонстрации сортировки
print("\nПервые 5 результатов (наименьшие I_e):")
print(f"{'vardelta':<10} {'r_m':<8} {'B':<8} {'omega (кг)':<15} {'I_e (МПа·с)':<15}")
print("-" * 85)
for i in range(5):
    r = results_sorted[i]
    print(f"{r['vardelta']:<10} {r['r_m']:<8.1f} {r['B']:<8} {r['omega']:<15.4f} {r['I_e']:<15.4f}")

print("\nПоследние 5 результатов (наибольшие I_e):")
print(f"{'vardelta':<10} {'r_m':<8} {'B':<8} {'omega (кг)':<15} {'I_e (МПа·с)':<15}")
print("-" * 85)
for i in range(-5, 0):
    r = results_sorted[i]
    print(f"{r['vardelta']:<10} {r['r_m']:<8.1f} {r['B']:<8} {r['omega']:<15.4f} {r['I_e']:<15.4f}")