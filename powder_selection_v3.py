import math
from math import *
import itertools
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from baza_powder import powders_db

# Классификация порохов
def classify_powder(powder_name):
    """Классифицирует порох по первому символу названия"""
    first_char = powder_name.strip()[0]
    if first_char.isalpha():  # начинаются с буквы - баллиститный
        return 'ballistic'
    else:  # начинаются с цифры или дроби - пироксилиновый
        return 'pyroxylin'

# Разделяем базу порохов по типам
pyroxylin_powders = {}
ballistic_powders = {}

for name, data in powders_db.items():
    powder_type = classify_powder(name)
    
    # Определяем B_value в зависимости от z_e
    if data['z_e'] == 1:
        B_value = 4.5  # одноканальный порох
    else:
        B_value = 2.0  # многоканальный порох
    
    # Добавляем B_value в данные пороха
    data['B_value'] = B_value
    
    if powder_type == 'pyroxylin':
        pyroxylin_powders[name] = data
    else:
        ballistic_powders[name] = data

print(f"Пироксилиновые пороха: {len(pyroxylin_powders)} шт.")
print(f"Баллиститные пороха: {len(ballistic_powders)} шт.")

# ПАРАМЕТРЫ ДЛЯ ДВУХ ТИПОВ УСЛОВНЫХ ПОРОХОВ
# Баллиститный порох (ДГ-3)
ballistic_params = {
    'name': 'Баллиститный (ДГ-3)',
    'f': 0.968e6,  # Дж/кг
    'k': 1.25,
    'b': 1.113e-3,  # м³/кг
    'delta': 1520,  # кг/м³
    'z_e': 1,
    'kappa': 1,
    'lambda_val': 0,
    'K_f': 0.0004,
    'K_l': 0.0022
}

# Пироксилиновый порох
pyroxylin_params = {
    'name': 'Пироксилиновый',
    'f': 1.0e6,  # Дж/кг
    'k': 1.23,
    'b': 1.0e-3,  # м³/кг
    'delta': 1600,  # кг/м³
    'z_e': 1,
    'kappa': 1,
    'lambda_val': 0,
    'K_f': 0.0003,
    'K_l': 0.0016
}

# Общие параметры для обоих случаев
v_pm = 950  # дульная скорость
d = 0.085
p_ign = 1e6
q = 5  # масса снаряда, кг 
phi_1 = 1.02
omega_ign = 0.01
S = pi * d**2 / 4

# Диапазоны варьирования (для пушек высокой мощности)
vardelta_values = list(range(650, 781, 10))  # кг/м³
eta_rm_values = list(np.arange(0.1, 0.6, 0.1))  # r_m от 0.1 до 0.5
B_values = [4.5, 2.0]  # B для одноканальных и многоканальных порохов

# Максимально возможное значение импульса
MAX_IMPULSE = 2.71

def I_e(vardelta, r_m, B, params):
    """Расчет массы пороха и импульса для заданных параметров"""
    f = params['f']
    k = params['k']
    b_val = params['b']
    delta = params['delta']
    
    zeta = (p_ign / f) * (1 / vardelta - 1 / delta) * 1 / (1 + (b_val * p_ign) / f)  
    
    omega = (phi_1 * q) / (((2 * f) / ((k - 1) * v_pm**2)) * r_m - (zeta + 1) / 3)
    
    phi = phi_1 * (1 / (3 * q)) * (omega_ign + omega)
    
    I_e_val = (math.sqrt(f * omega * phi * q * B) / S) / 10**6
    
    return omega, I_e_val

def find_matching_powders(target_impulse, powders_db, tolerance=0.1):
    """Поиск порохов с подходящим импульсом в указанной базе"""
    matching_powders = []
    
    for powder_name, powder_data in powders_db.items():
        powder_impulse = powder_data['I_e']
        # Пропускаем пороха с импульсом больше максимального
        if powder_impulse > MAX_IMPULSE:
            continue
            
        difference = powder_impulse - target_impulse
        difference_percent = (difference / target_impulse) * 100 if target_impulse != 0 else 0
        
        if abs(difference) <= tolerance:
            matching_powders.append((powder_name, powder_impulse, powder_data, difference, difference_percent))
    
    matching_powders.sort(key=lambda x: abs(x[3]))  # Сортировка по абсолютному отклонению
    return matching_powders

def vary_parameters_for_powder_type(params, powder_type_name, powders_db):
    """Варьирование параметров для конкретного типа пороха"""
    results = []
    
    # Создаем комбинации только с vardelta, r_m и B
    combinations = list(itertools.product(vardelta_values, eta_rm_values, B_values))
    
    print(f"\n{'='*90}")
    print(f"РАСЧЕТ ДЛЯ {powder_type_name.upper()}")
    print(f"{'='*90}")
    print(f"{'vardelta':<10} {'r_m':<8} {'B':<8} {'omega (кг)':<15} {'I_e (МПа·с)':<15} {'Статус':<10}")
    print("-" * 90)
    
    valid_combinations = 0
    exceeded_combinations = 0
    
    for vardelta, r_m, B in combinations:
        omega, impulse = I_e(vardelta, r_m, B, params)
        
        # Пропускаем комбинации с импульсом больше максимального
        if impulse > MAX_IMPULSE:
            status = "ПРЕВЫШЕН"
            print(f"{vardelta:<10} {r_m:<8.1f} {B:<8} {omega:<15.4f} {impulse:<15.4f} {status:<10}")
            exceeded_combinations += 1
            continue
        
        results.append({
            'vardelta': vardelta,
            'r_m': r_m,
            'B': B,
            'omega': omega,
            'I_e': impulse
        })
        
        status = "OK"
        print(f"{vardelta:<10} {r_m:<8.1f} {B:<8} {omega:<15.4f} {impulse:<15.4f} {status:<10}")
        valid_combinations += 1
    
    # Сводка по рассчитанным импульсам
    if results:
        calculated_impulses = [r['I_e'] for r in results]
        print(f"\n📊 СВОДКА ПО РАССЧИТАННЫМ ИМПУЛЬСАМ:")
        print(f"   Минимальный импульс: {min(calculated_impulses):.4f} МПа·с")
        print(f"   Максимальный импульс: {max(calculated_impulses):.4f} МПа·с")
        print(f"   Средний импульс: {sum(calculated_impulses)/len(calculated_impulses):.4f} МПа·с")
        print(f"   Диапазон: {min(calculated_impulses):.4f} - {max(calculated_impulses):.4f} МПа·с")
        print(f"   Допустимый максимум: {MAX_IMPULSE} МПа·с")
    else:
        print(f"\n⚠️  Нет допустимых комбинаций - все превышают максимальный импульс {MAX_IMPULSE} МПа·с")
    
    print(f"Всего комбинаций: {len(combinations)}")
    print(f"Валидных комбинаций: {valid_combinations}")
    print(f"Комбинаций с превышением: {exceeded_combinations}")
    return results

def print_detailed_matching_analysis(results, powders_db, powder_type_name):
    """Детальный анализ соответствия порохов рассчитанным значениям"""
    if not results:
        print(f"⚠️  Нет данных для анализа {powder_type_name}")
        return
        
    calculated_impulses = [r['I_e'] for r in results]
    avg_impulse = sum(calculated_impulses) / len(calculated_impulses)
    
    print(f"\n🔍 ДЕТАЛЬНЫЙ АНАЛИЗ ДЛЯ {powder_type_name.upper()}:")
    print(f"   Средний рассчитанный импульс: {avg_impulse:.4f} МПа·с")
    
    # Анализ лучших совпадений (только пороха с импульсом <= MAX_IMPULSE)
    best_matches = []
    for powder_name, powder_data in powders_db.items():
        powder_impulse = powder_data['I_e']
        if powder_impulse > MAX_IMPULSE:
            continue
            
        difference = abs(powder_impulse - avg_impulse)
        difference_percent = (difference / avg_impulse) * 100
        best_matches.append((powder_name, powder_impulse, difference, difference_percent, powder_data['B_value'], powder_data['z_e']))
    
    # Сортируем по близости к среднему
    best_matches.sort(key=lambda x: x[2])
    
    print(f"\n🎯 ТОП-10 самых близких порохов к среднему значению:")
    print(f"{'Марка':<25} {'I_e (МПа·с)':<12} {'Разница':<12} {'%':<8} {'B':<6} {'z_e':<6}")
    print("-" * 75)
    for i, (name, impulse, diff, diff_percent, B_val, z_e) in enumerate(best_matches[:10]):
        print(f"{i+1:2}. {name:<22} {impulse:<12.3f} {diff:<12.3f} {diff_percent:+.1f}% {B_val:<6} {z_e:<6}")

def create_approved_powders_file(results, powders_db, params, powder_type, best_count=10):
    """Создает файл с одобренными порохами для конкретного типа"""
    
    if not results:
        print(f"⚠️  Нет результатов для создания файла {powder_type}")
        return None
    
    calculated_impulses = [r['I_e'] for r in results]
    avg_impulse = sum(calculated_impulses) / len(calculated_impulses)
    
    min_calc_impulse = min(calculated_impulses)
    max_calc_impulse = max(calculated_impulses)
    
    suitable_powders = {}
    
    # Собираем подходящие пороха (только с импульсом <= MAX_IMPULSE)
    for powder_name, powder_data in powders_db.items():
        powder_impulse = powder_data['I_e']
        if powder_impulse > MAX_IMPULSE:
            continue
            
        if min_calc_impulse <= powder_impulse <= max_calc_impulse:
            suitable_powders[powder_name] = powder_data
    
    # Определяем лучшие пороха по минимальному отклонению от среднего
    best_powders = set()
    if suitable_powders:
        # Сортируем пороха по близости к среднему импульсу
        sorted_powders = sorted(
            suitable_powders.items(),
            key=lambda x: abs(x[1]['I_e'] - avg_impulse)
        )
        # Берем топ-5 самых близких
        top_count = min(5, len(sorted_powders))
        best_powders = {name for name, _ in sorted_powders[:top_count]}
    
    # Создаем содержимое файла
    filename = f"approved_{powder_type}.py"
    
    file_content = f"""# Файл с одобренными {powder_type} порохами
# Сгенерировано автоматически на основе варьирования параметров
# Параметры условного пороха: {params['name']}
# f={params['f']:.3e} Дж/кг, k={params['k']:.3f}, b={params['b']:.3e} м³/кг, delta={params['delta']} кг/м³
# Максимальный импульс: {MAX_IMPULSE} МПа·с

# ДИАПАЗОН РАССЧИТАННЫХ ИМПУЛЬСОВ: {min_calc_impulse:.4f} - {max_calc_impulse:.4f} МПа·с
# СРЕДНИЙ ИМПУЛЬС: {avg_impulse:.4f} МПа·с

approved_powders = {{
"""
    
    for powder_name in sorted(suitable_powders.keys()):
        powder_data = suitable_powders[powder_name]
        powder_impulse = powder_data['I_e']
        difference = powder_impulse - avg_impulse
        difference_percent = (difference / avg_impulse) * 100
        
        if powder_name in best_powders:
            file_content += f"    # ★ ЛУЧШИЙ ВАРИАНТ - {powder_name} (отклонение: {difference:+.3f} МПа·с, {difference_percent:+.1f}%)\n"
        else:
            file_content += f"    # {powder_name} (отклонение от среднего: {difference:+.3f} МПа·с, {difference_percent:+.1f}%)\n"
        
        file_content += f"    '{powder_name}': {{\n"
        for key, value in powder_data.items():
            if key == 'B_value':  # Не сохраняем вычисленное поле
                continue
            if isinstance(value, float):
                file_content += f"        '{key}': {value:.3f},\n"
            else:
                file_content += f"        '{key}': {value},\n"
        file_content += "    },\n\n"
    
    file_content += "}\n\n"
    
    # Добавляем информацию о лучших комбинациях параметров
    file_content += "# ЛУЧШИЕ КОМБИНАЦИИ ПАРАМЕТРОВ:\n"
    file_content += "#" * 70 + "\n"
    
    # Находим лучшие комбинации (ближайшие к среднему)
    best_combinations = sorted(results, key=lambda x: abs(x['I_e'] - avg_impulse))[:5]
    
    for i, combo in enumerate(best_combinations, 1):
        file_content += f"# Комбинация #{i}: vardelta={combo['vardelta']}, r_m={combo['r_m']:.1f}, B={combo['B']}\n"
        file_content += f"# Результат: omega={combo['omega']:.4f} кг, I_e={combo['I_e']:.4f} МПа·с\n"
        
        # Находим пороха, подходящие для этой комбинации
        matching_powders = []
        for powder_name, powder_data in powders_db.items():
            if powder_data['I_e'] > MAX_IMPULSE:
                continue
            if powder_data['B_value'] == combo['B']:  # Только пороха с подходящим B
                difference = abs(powder_data['I_e'] - combo['I_e'])
                if difference <= 0.15:
                    matching_powders.append((powder_name, powder_data['I_e'], difference))
        
        if matching_powders:
            matching_powders.sort(key=lambda x: x[2])
            matching_str = ', '.join([f'{name}({imp:.3f})' for name, imp, _ in matching_powders[:3]])
            file_content += f"# Подходящие пороха: {matching_str}\n"
        file_content += "#" * 70 + "\n"
    
    # Записываем файл
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(file_content)
    
    print(f"\n✅ Файл '{filename}' успешно создан!")
    print(f"📊 Всего одобрено порохов: {len(suitable_powders)}")
    print(f"🏆 Лучших порохов выделено: {len(best_powders)}")
    
    # Детальный анализ подходящих порохов
    if suitable_powders:
        print(f"\n📈 АНАЛИЗ ПОДХОДЯЩИХ ПОРОХОВ:")
        print(f"{'Марка':<25} {'I_e (МПа·с)':<12} {'Отклонение':<15} {'%':<8} {'B':<6} {'z_e':<6}")
        print("-" * 75)
        for powder_name in sorted(suitable_powders.keys()):
            powder_data = suitable_powders[powder_name]
            powder_impulse = powder_data['I_e']
            difference = powder_impulse - avg_impulse
            difference_percent = (difference / avg_impulse) * 100
            
            marker = "★ " if powder_name in best_powders else "  "
            print(f"{marker}{powder_name:<23} {powder_impulse:<12.3f} {difference:+.3f} МПа·с   {difference_percent:+.1f}% {powder_data['B_value']:<6} {powder_data['z_e']:<6}")
    
    return filename

# ЗАПУСК РАСЧЕТОВ ДЛЯ ОБОИХ ТИПОВ ПОРОХОВ

# 1. Расчет для баллиститных порохов
print(f"\n{'='*90}")
print("🚀 ЗАПУСК РАСЧЕТА ДЛЯ БАЛЛИСТИТНЫХ ПОРОХОВ")
print(f"{'='*90}")
ballistic_results = vary_parameters_for_powder_type(
    ballistic_params, 'БАЛЛИСТИТНЫХ', ballistic_powders
)
if ballistic_results:
    print_detailed_matching_analysis(ballistic_results, ballistic_powders, "БАЛЛИСТИТНЫХ")
    ballistic_file = create_approved_powders_file(
        ballistic_results, ballistic_powders, ballistic_params, 'ballistic'
    )
else:
    ballistic_file = None
    print("❌ Нет допустимых результатов для баллиститных порохов")

# 2. Расчет для пироксилиновых порохов
print(f"\n{'='*90}")
print("🚀 ЗАПУСК РАСЧЕТА ДЛЯ ПИРОКСИЛИНОВЫХ ПОРОХОВ")
print(f"{'='*90}")
pyroxylin_results = vary_parameters_for_powder_type(
    pyroxylin_params, 'ПИРОКСИЛИНОВЫХ', pyroxylin_powders
)
if pyroxylin_results:
    print_detailed_matching_analysis(pyroxylin_results, pyroxylin_powders, "ПИРОКСИЛИНОВЫХ")
    pyroxylin_file = create_approved_powders_file(
        pyroxylin_results, pyroxylin_powders, pyroxylin_params, 'pyroxylin'
    )
else:
    pyroxylin_file = None
    print("❌ Нет допустимых результатов для пироксилиновых порохов")

print(f"\n{'='*90}")
print("🎉 РАСЧЕТЫ ЗАВЕРШЕНЫ!")
if ballistic_file or pyroxylin_file:
    print("📁 Созданы файлы:")
    if ballistic_file:
        print(f"   • {ballistic_file}")
    if pyroxylin_file:
        print(f"   • {pyroxylin_file}")
else:
    print("❌ Файлы не созданы - нет допустимых результатов")
print(f"{'='*90}")


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def create_scatter_plot(results, powders_db, params, powder_type):
    """Создает диаграмму рассеяния для сравнения импульсов"""
    
    if not results:
        print(f"Нет данных для визуализации {powder_type}")
        return
    
    # Подготовка данных
    calculated_impulses = [r['I_e'] for r in results]
    avg_impulse = np.mean(calculated_impulses)
    
    # Создаем DataFrame для порохов
    powder_data = []
    for name, data in powders_db.items():
        if data['I_e'] > 2.71:  # Пропускаем превышающие максимум
            continue
        powder_data.append({
            'name': name,
            'powder_impulse': data['I_e'],
            'type': 'ballistic' if classify_powder(name) == 'ballistic' else 'pyroxylin',
            'B_value': data['B_value'],
            'z_e': data['z_e']
        })
    
    df_powders = pd.DataFrame(powder_data)
    
    # Создаем график
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # График 1: Сравнение с расчетными значениями
    ax1.scatter(calculated_impulses, [avg_impulse] * len(calculated_impulses), 
               alpha=0.6, color='blue', label='Расчетные значения', s=50)
    
    colors = {'ballistic': 'red', 'pyroxylin': 'green'}
    for powder_type_val in ['ballistic', 'pyroxylin']:
        mask = df_powders['type'] == powder_type_val
        ax1.scatter(df_powders[mask]['powder_impulse'], 
                   [avg_impulse] * len(df_powders[mask]),
                   alpha=0.7, color=colors[powder_type_val], 
                   label=f'Пороха ({powder_type_val})', s=60)
    
    ax1.axvline(x=avg_impulse, color='black', linestyle='--', alpha=0.7, label=f'Среднее: {avg_impulse:.3f}')
    ax1.axvline(x=2.71, color='red', linestyle=':', alpha=0.7, label='Максимум: 2.71')
    ax1.set_xlabel('Импульс I_e (МПа·с)')
    ax1.set_ylabel('Средний расчетный импульс')
    ax1.set_title(f'Сравнение импульсов - {powder_type}\n{params["name"]}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # График 2: Отклонения от среднего
    deviations = df_powders['powder_impulse'] - avg_impulse
    colors_dev = [colors[t] for t in df_powders['type']]
    
    bars = ax2.barh(range(len(df_powders)), deviations, color=colors_dev, alpha=0.7)
    ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    ax2.set_xlabel('Отклонение от среднего (МПа·с)')
    ax2.set_ylabel('Марка пороха')
    ax2.set_title('Отклонения импульсов порохов от среднего')
    ax2.set_yticks(range(len(df_powders)))
    ax2.set_yticklabels(df_powders['name'], fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Добавляем значения на столбцы
    for i, (bar, deviation) in enumerate(zip(bars, deviations)):
        ax2.text(deviation + (0.01 if deviation >= 0 else -0.05), 
                bar.get_y() + bar.get_height()/2,
                f'{deviation:+.3f}', 
                va='center', ha='left' if deviation >= 0 else 'right',
                fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'scatter_plot_{powder_type}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

# ВСТАВЬ ЭТО ПОСЛЕ ТВОЕГО ОСНОВНОГО КОДА:

# После создания approved файлов добавляем визуализацию
print(f"\n{'='*90}")
print("📊 СОЗДАНИЕ ВИЗУАЛИЗАЦИИ")
print(f"{'='*90}")

# Для баллиститных порохов
if ballistic_results:
    print("Создание диаграммы рассеяния для баллиститных порохов...")
    create_scatter_plot(ballistic_results, ballistic_powders, ballistic_params, 'ballistic')
else:
    print("❌ Нет данных для визуализации баллиститных порохов")

# Для пироксилиновых порохов
if pyroxylin_results:
    print("Создание диаграммы рассеяния для пироксилиновых порохов...")
    create_scatter_plot(pyroxylin_results, pyroxylin_powders, pyroxylin_params, 'pyroxylin')
else:
    print("❌ Нет данных для визуализации пироксилиновых порохов")

print(f"\n{'='*90}")
print("✅ ВИЗУАЛИЗАЦИЯ ЗАВЕРШЕНА")
print(f"{'='*90}")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def create_sankey_diagram_png(results, powders_db, params, powder_type):
    """Создает упрощенную Sankey диаграмму в формате PNG"""
    
    if not results:
        print(f"Нет данных для Sankey диаграммы {powder_type}")
        return
    
    # Упрощенная группировка
    vardelta_bins = ['650-700', '701-750', '751-780']
    rm_bins = ['0.1-0.3', '0.4-0.5']
    impulse_bins = ['<1.5', '1.5-2.71']
    
    # Создаем упрощенную визуализацию с matplotlib
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Упрощенные данные для отображения
    categories = []
    flows = []
    
    # Подсчитываем основные потоки
    for vd_bin in vardelta_bins:
        for rm_bin in rm_bins:
            count = 0
            for r in results:
                vd_ok = False
                rm_ok = False
                
                if vd_bin == '650-700' and 650 <= r['vardelta'] <= 700:
                    vd_ok = True
                elif vd_bin == '701-750' and 701 <= r['vardelta'] <= 750:
                    vd_ok = True
                elif vd_bin == '751-780' and 751 <= r['vardelta'] <= 780:
                    vd_ok = True
                
                if rm_bin == '0.1-0.3' and 0.1 <= r['r_m'] <= 0.3:
                    rm_ok = True
                elif rm_bin == '0.4-0.5' and 0.4 <= r['r_m'] <= 0.5:
                    rm_ok = True
                
                if vd_ok and rm_ok:
                    count += 1
            
            if count > 0:
                categories.append(f"{vd_bin}→{rm_bin}")
                flows.append(count)
    
    # Создаем горизонтальную столбчатую диаграмму
    y_pos = np.arange(len(categories))
    
    bars = ax.barh(y_pos, flows, alpha=0.7, color='lightblue', edgecolor='black')
    
    # Добавляем значения на столбцы
    for i, (bar, flow) in enumerate(zip(bars, flows)):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f'{flow} комб.', va='center', ha='left', fontsize=9)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=10)
    ax.set_xlabel('Количество комбинаций')
    ax.set_title(f'Sankey-подобная диаграмма - {powder_type}\n{params["name"]}', fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Добавляем информацию о порохах
    calculated_impulses = [r['I_e'] for r in results]
    min_impulse = min(calculated_impulses)
    max_impulse = max(calculated_impulses)
    
    suitable_powders = []
    for powder_name, data in powders_db.items():
        if data['I_e'] > 2.71:
            continue
        if min_impulse <= data['I_e'] <= max_impulse:
            suitable_powders.append(powder_name)
    
    # Ограничиваем количество для читаемости
    suitable_powders = suitable_powders[:8]
    
    # Добавляем информацию о порохах как текст
    powder_text = "Подходящие пороха:\n" + "\n".join([f"• {name}" for name in suitable_powders])
    ax.text(0.02, 0.98, powder_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
    
    # Статистика
    stats_text = f"Статистика:\nКомбинаций: {len(results)}\nДиапазон I_e: {min_impulse:.2f}-{max_impulse:.2f}\nСреднее: {np.mean(calculated_impulses):.2f}"
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(f'sankey_diagram_{powder_type}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Sankey диаграмма (PNG) сохранена как sankey_diagram_{powder_type}.png")
    
    # Дополнительная информация в консоль
    print(f"\n📋 СВОДКА ДЛЯ {powder_type.upper()}:")
    print(f"   Всего комбинаций: {len(results)}")
    print(f"   Диапазон импульсов: {min_impulse:.3f} - {max_impulse:.3f} МПа·с")
    print(f"   Подходящих порохов: {len(suitable_powders)}")
    if suitable_powders:
        print(f"   Лучшие пороха: {', '.join(suitable_powders[:5])}")
    
    return fig

# ВСТАВЬ ЭТО ПОСЛЕ ВИЗУАЛИЗАЦИИ №1:

print(f"\n{'='*90}")
print("📊 СОЗДАНИЕ SANKEY ДИАГРАММ (PNG)")
print(f"{'='*90}")

# Для баллиститных порохов
if ballistic_results:
    print("Создание Sankey диаграммы для баллиститных порохов...")
    create_sankey_diagram_png(ballistic_results, ballistic_powders, ballistic_params, 'ballistic')
else:
    print("❌ Нет данных для Sankey диаграммы баллиститных порохов")

# Для пироксилиновых порохов
if pyroxylin_results:
    print("Создание Sankey диаграммы для пироксилиновых порохов...")
    create_sankey_diagram_png(pyroxylin_results, pyroxylin_powders, pyroxylin_params, 'pyroxylin')
else:
    print("❌ Нет данных для Sankey диаграммы пироксилиновых порохов")

print(f"\n{'='*90}")
print("✅ SANKEY ДИАГРАММЫ СОЗДАНЫ")
print("📁 Файлы сохранены в формате PNG")
print(f"{'='*90}")






