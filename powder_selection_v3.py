import math
from math import *
import itertools
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

# Разделяем базу порохов по типам и определяем B_values для каждого пороха
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
v_pm = 900  # дульная скорость
d = 0.1524
p_ign = 1e6
q = 54  # масса снаряда, кг 
phi_1 = 1.02
omega_ign = 0.01
S = pi * d**2 / 4

# Диапазоны варьирования (для пушек высокой мощности)
vardelta_values = list(range(650, 781, 10))  # кг/м³
eta_rm_values = list(np.arange(0.1, 0.6, 0.1))  # r_m от 0.1 до 0.5
# B_values больше не варьируем - используем индивидуальные для каждого пороха

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
    
    # Создаем комбинации только с vardelta и r_m, B берем из данных пороха
    combinations = list(itertools.product(vardelta_values, eta_rm_values))
    
    print(f"\n{'='*90}")
    print(f"РАСЧЕТ ДЛЯ {powder_type_name.upper()}")
    print(f"{'='*90}")
    print(f"{'vardelta':<10} {'r_m':<8} {'B':<8} {'omega (кг)':<15} {'I_e (МПа·с)':<15} {'Статус':<10}")
    print("-" * 90)
    
    valid_combinations = 0
    
    for vardelta, r_m in combinations:
        # Для каждого пороха в базе рассчитываем с его B_value
        for powder_name, powder_data in powders_db.items():
            B = powder_data['B_value']
            omega, impulse = I_e(vardelta, r_m, B, params)
            
            # Пропускаем комбинации с импульсом больше максимального
            if impulse > MAX_IMPULSE:
                status = "ПРЕВЫШЕН"
                print(f"{vardelta:<10} {r_m:<8.1f} {B:<8} {omega:<15.4f} {impulse:<15.4f} {status:<10}")
                continue
            
            results.append({
                'vardelta': vardelta,
                'r_m': r_m,
                'B': B,
                'omega': omega,
                'I_e': impulse,
                'powder_name': powder_name
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
    
    print(f"Всего валидных комбинаций: {valid_combinations}")
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
        best_matches.append((powder_name, powder_impulse, difference, difference_percent))
    
    # Сортируем по близости к среднему
    best_matches.sort(key=lambda x: x[2])
    
    print(f"\n🎯 ТОП-10 самых близких порохов к среднему значению:")
    print(f"{'Марка':<25} {'I_e (МПа·с)':<12} {'Разница':<12} {'%':<8} {'B':<6} {'z_e':<6}")
    print("-" * 75)
    for i, (name, impulse, diff, diff_percent) in enumerate(best_matches[:10]):
        powder_data = powders_db[name]
        print(f"{i+1:2}. {name:<22} {impulse:<12.3f} {diff:<12.3f} {diff_percent:+.1f}% {powder_data['B_value']:<6} {powder_data['z_e']:<6}")

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