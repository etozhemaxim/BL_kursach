import math
import itertools
import numpy as np
import matplotlib.pyplot as plt

# Импортируем базу данных порохов из файла baza_powder.py
from baza_powder import powders_db

# Исключенные пороха
EXCLUDED_POWDERS = {
    '100/70', '180/57 Ш3 БП', '100/56', '180/57', '152/57 БП', 
    '152/57 Ш', '130/50', '152/57', '180/57 БП', '180/60'
}

# Константы для разных типов порохов
PYROXYLIN = {
    'f': 1.0e6,    # Дж/кг
    'k': 1.23,     # -
    'b': 1.0e-3,   # м³/кг
    'delta': 1600  # кг/м³
}

BALLISTITE = {
    'f': 0.968e6,  # Дж/кг
    'k': 1.25,     # -
    'b': 1.113e-3, # м³/кг  
    'delta': 1520  # кг/м³
}

# Диапазоны варьирования параметров
vardelta_values = list(range(550, 700, 10))  # кг/м³
eta_rm_values = list(np.arange(0.1, 0.6, 0.1))  # r_m
B_values = [4.5, 2.0]  # B_1 (трубчатый) и B_7 (семиканальный)

# Фиксированные параметры
v_pm = 950  # м/с (дульная скорость)
d = 0.085   # м (калибр)
q = 5.0     # кг (масса снаряда)
phi_1 = 1.02  # коэффициент фиктивности
omega_ign = 0.01  # кг (масса воспламенителя)
p_ign = 1e6  # Па
S = math.pi * d**2 / 4  # м² (площадь поперечного сечения)

def get_powder_type(powder_name, powder_data):
    """Определение типа и вида пороха"""
    # Проверяем, исключен ли порох
    if powder_name in EXCLUDED_POWDERS:
        return None, None
    
    z_e = powder_data['z_e']
    
    # Определяем тип пороха по началу названия
    if powder_name[0].isalpha():  # Начинается с буквы
        powder_type = 'ballistite'
    else:  # Начинается с цифры или дроби
        powder_type = 'pyroxylin'
    
    # Определяем вид пороха по z_e
    if z_e == 1:
        powder_kind = 'single_channel'
    else:
        powder_kind = 'multi_channel'
    
    return powder_type, powder_kind

def calculate_powder_impulse(powder_type, vardelta, r_m, B):
    """Расчет импульса пороха для заданных параметров"""
    f = powder_type['f']
    k = powder_type['k']
    b = powder_type['b']
    delta = powder_type['delta']
    
    # Расчет коэффициентов
    zeta = (p_ign / f) * (1 / vardelta - 1 / delta) * 1 / (1 + (b * p_ign) / f)  
    
    # Расчет массы пороха
    denominator = ((2 * f) / ((k - 1) * v_pm**2)) * r_m - (zeta + 1) / 3
    if denominator <= 0:
        return None, None  # Некорректные параметры
    
    omega = (phi_1 * q) / denominator
    
    # Расчет коэффициента фиктивности
    phi = phi_1 + (1 / (3 * q)) * (omega_ign + omega)
    
    # Расчет импульса (в МПа·с)
    I_e = (math.sqrt(f * omega * phi * q * B) / S) / 10**6
    
    # Ограничение по максимальному импульсу
    if I_e > 2.71:
        return None, None
    
    return omega, I_e

def calculate_all_variations(powder_type, powder_name):
    """Расчет всех вариаций для заданного типа пороха"""
    results_tubular = []    # Для B = 4.5 (трубчатые)
    results_seven_channel = []  # Для B = 2.0 (семиканальные)
    
    # Создаем все комбинации параметров
    combinations = list(itertools.product(vardelta_values, eta_rm_values, B_values))
    
    valid_combinations = 0
    
    for vardelta, r_m, B in combinations:
        omega, impulse = calculate_powder_impulse(powder_type, vardelta, r_m, B)
        
        if omega is not None and impulse is not None:
            result = {
                'vardelta': vardelta,
                'r_m': r_m,
                'B': B,
                'omega': omega,
                'I_e': impulse
            }
            
            # Разделяем по типу пороха (B)
            if B == 4.5:
                results_tubular.append(result)
            else:  # B == 2.0
                results_seven_channel.append(result)
            
            valid_combinations += 1
    
    print(f"  Корректных комбинаций: {valid_combinations}")
    print(f"  Трубчатых (B=4.5): {len(results_tubular)}")
    print(f"  Семиканальных (B=2.0): {len(results_seven_channel)}")
    
    return results_tubular, results_seven_channel

def calculate_statistics(results):
    """Расчет статистики для массива результатов"""
    if not results:
        return 0.0, 0.0, 0.0
    
    impulses = [r['I_e'] for r in results]
    avg_impulse = sum(impulses) / len(impulses)
    min_impulse = min(impulses)
    max_impulse = max(impulses)
    
    return avg_impulse, min_impulse, max_impulse

def analyze_powders_database():
    """Анализ базы данных порохов"""
    # Сначала рассчитываем средние значения для условных порохов
    print("Расчет средних значений для условных порохов...")
    
    # Расчет для пироксилинового пороха
    print("Пироксилиновый порох:")
    pyro_tubular, pyro_seven = calculate_all_variations(PYROXYLIN, "пироксилинового")
    pyro_tubular_avg, _, _ = calculate_statistics(pyro_tubular)
    pyro_seven_avg, _, _ = calculate_statistics(pyro_seven)
    
    # Расчет для баллиститного пороха  
    print("Баллиститный порох:")
    ball_tubular, ball_seven = calculate_all_variations(BALLISTITE, "баллиститного")
    ball_tubular_avg, _, _ = calculate_statistics(ball_tubular)
    ball_seven_avg, _, _ = calculate_statistics(ball_seven)
    
    print("\nСредние значения условных порохов:")
    print(f"  Пироксилиновые трубчатые: {pyro_tubular_avg:.4f} МПа·с")
    print(f"  Пироксилиновые семиканальные: {pyro_seven_avg:.4f} МПа·с")
    print(f"  Баллиститные трубчатые: {ball_tubular_avg:.4f} МПа·с")
    print(f"  Баллиститные семиканальные: {ball_seven_avg:.4f} МПа·с")
    
    # Массивы для хранения результатов анализа
    results = {
        'ballistite_single_channel': [],   # Баллиститные одноканальные
        'ballistite_multi_channel': [],    # Баллиститные многоканальные
        'pyroxylin_single_channel': [],    # Пироксилиновые одноканальные
        'pyroxylin_multi_channel': []      # Пироксилиновые многоканальные
    }
    
    print(f"\nАнализ базы данных порохов ({len(powders_db)} порохов)...")
    print("=" * 90)
    print(f"{'Название':<20} {'Тип':<12} {'Вид':<15} {'I_e из БД':<10} {'Среднее':<10} {'Отклонение %':<12}")
    print("-" * 90)
    
    analyzed_count = 0
    excluded_count = 0
    
    for powder_name, powder_data in powders_db.items():
        # Определяем тип и вид пороха
        powder_type, powder_kind = get_powder_type(powder_name, powder_data)
        
        if powder_type is None:  # Порох исключен
            excluded_count += 1
            continue
        
        # Получаем импульс из базы данных
        I_e_db = powder_data['I_e']
        
        # Определяем среднее значение для сравнения
        if powder_type == 'ballistite' and powder_kind == 'single_channel':
            avg_I_e = ball_tubular_avg
            category = 'ballistite_single_channel'
            powder_type_ru = 'баллистит'
            powder_kind_ru = 'одноканал'
        elif powder_type == 'ballistite' and powder_kind == 'multi_channel':
            avg_I_e = ball_seven_avg
            category = 'ballistite_multi_channel'
            powder_type_ru = 'баллистит'
            powder_kind_ru = 'многоканал'
        elif powder_type == 'pyroxylin' and powder_kind == 'single_channel':
            avg_I_e = pyro_tubular_avg
            category = 'pyroxylin_single_channel'
            powder_type_ru = 'пироксилин'
            powder_kind_ru = 'одноканал'
        else:  # pyroxylin multi_channel
            avg_I_e = pyro_seven_avg
            category = 'pyroxylin_multi_channel'
            powder_type_ru = 'пироксилин'
            powder_kind_ru = 'многоканал'
        
        # Вычисляем отклонение в процентах
        if avg_I_e > 0:
            deviation_percent = ((I_e_db - avg_I_e) / avg_I_e) * 100
        else:
            deviation_percent = 0
        
        # Записываем в массив
        results[category].append({
            'name': powder_name,
            'I_e': I_e_db,
            'deviation_percent': deviation_percent,
            'avg_I_e': avg_I_e
        })
        
        analyzed_count += 1
        print(f"{powder_name:<20} {powder_type_ru:<12} {powder_kind_ru:<15} {I_e_db:<10.3f} {avg_I_e:<10.3f} {deviation_percent:<12.2f}")
    
    print("=" * 90)
    print(f"Проанализировано порохов: {analyzed_count}")
    print(f"Исключено порохов: {excluded_count}")
    
    return results, {
        'ballistite_single_channel_avg': ball_tubular_avg,
        'ballistite_multi_channel_avg': ball_seven_avg,
        'pyroxylin_single_channel_avg': pyro_tubular_avg,
        'pyroxylin_multi_channel_avg': pyro_seven_avg
    }

def plot_results(results, avg_values):
    """Построение графиков отклонений"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Отклонение импульсов порохов от средних значений по категориям', fontsize=16, fontweight='bold')
    
    categories = [
        ('ballistite_single_channel', 'Баллиститные одноканальные', 'red', axes[0, 0]),
        ('ballistite_multi_channel', 'Баллиститные многоканальные', 'blue', axes[0, 1]),
        ('pyroxylin_single_channel', 'Пироксилиновые одноканальные', 'green', axes[1, 0]),
        ('pyroxylin_multi_channel', 'Пироксилиновые многоканальные', 'orange', axes[1, 1])
    ]
    
    for category, title, color, ax in categories:
        data = results[category]
        
        if not data:
            ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(title, fontsize=12, fontweight='bold')
            continue
        
        # Извлекаем данные для графика
        deviations = [d['deviation_percent'] for d in data]
        impulses = [d['I_e'] for d in data]
        names = [d['name'] for d in data]
        
        # Строим график
        scatter = ax.scatter(deviations, impulses, c=color, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
        
        # Добавляем среднюю линию
        avg_key = f"{category}_avg"
        if avg_key in avg_values:
            avg_line = avg_values[avg_key]
            ax.axhline(y=avg_line, color='red', linestyle='--', alpha=0.7, label=f'Среднее: {avg_line:.3f}')
        
        # Настройки графика
        ax.set_xlabel('Отклонение от среднего (%)', fontsize=10)
        ax.set_ylabel('Импульс I_e (МПа·с)', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Добавляем аннотации для некоторых точек
        for i, (dev, imp, name) in enumerate(zip(deviations, impulses, names)):
            if i % 3 == 0:  # Аннотируем каждую 3-ю точку чтобы не загромождать
                ax.annotate(name, (dev, imp), xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
    # Дополнительная статистика по категориям
    print(f"\nСтатистика по категориям:")
    print("=" * 80)
    print(f"{'Категория':<30} {'Кол-во':<8} {'Среднее откл. %':<15} {'Мин откл. %':<12} {'Макс откл. %':<12}")
    print("-" * 80)
    
    for category, title, _, _ in categories:
        data = results[category]
        if data:
            deviations = [d['deviation_percent'] for d in data]
            avg_dev = sum(deviations) / len(deviations)
            min_dev = min(deviations)
            max_dev = max(deviations)
            print(f"{title:<30} {len(data):<8} {avg_dev:<15.2f} {min_dev:<12.2f} {max_dev:<12.2f}")
        else:
            print(f"{title:<30} {0:<8} {'-':<15} {'-':<12} {'-':<12}")

# Основная программа
def main():
    print("АНАЛИЗ БАЗЫ ДАННЫХ ПОРОХОВ")
    print("=" * 50)
    
    # Анализируем базу данных порохов
    results, avg_values = analyze_powders_database()
    
    # Строим графики
    plot_results(results, avg_values)
    
    return results, avg_values

# Запуск программы
if __name__ == "__main__":
    results, avg_values = main()