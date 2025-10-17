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
phi_1 = 1.04  # коэффициент фиктивности
omega_ign = 0.01  # кг (масса воспламенителя)
p_ign = 5e6  # Па
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
    # Баллиститные пороха всегда одноканальные (z_e = 1)
    if powder_type == 'ballistite':
        powder_kind = 'single_channel'
    else:
        # Пироксилиновые могут быть как одноканальные, так и многоканальные
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
    # Баллиститные пороха только трубчатые (одноканальные)
    ball_tubular, _ = calculate_all_variations(BALLISTITE, "баллиститного")
    ball_tubular_avg, _, _ = calculate_statistics(ball_tubular)
    ball_seven_avg = 0.0  # Баллиститных многоканальных не существует
    
    print("\nСредние значения условных порохов:")
    print(f"  Пироксилиновые трубчатые: {pyro_tubular_avg:.4f} МПа·с")
    print(f"  Пироксилиновые семиканальные: {pyro_seven_avg:.4f} МПа·с")
    print(f"  Баллиститные трубчатые: {ball_tubular_avg:.4f} МПа·с")
    print(f"  Баллиститные семиканальные: не существуют")
    
    # Массивы для хранения результатов анализа
    results = {
        'ballistite_single_channel': [],   # Баллиститные одноканальные
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
        if powder_type == 'ballistite':
            # Баллиститные пороха всегда одноканальные
            avg_I_e = ball_tubular_avg
            category = 'ballistite_single_channel'
            powder_type_ru = 'баллистит'
            powder_kind_ru = 'одноканал'
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
        
        # Определяем, является ли порох "лучшим" (отклонение в пределах [-30%;30%])
        is_best = -30 <= deviation_percent <= 30
        
        # Записываем в массив
        results[category].append({
            'name': powder_name,
            'I_e': I_e_db,
            'deviation_percent': deviation_percent,
            'avg_I_e': avg_I_e,
            'is_best': is_best  # Добавляем флаг "лучшего" пороха
        })
        
        analyzed_count += 1
        print(f"{powder_name:<20} {powder_type_ru:<12} {powder_kind_ru:<15} {I_e_db:<10.3f} {avg_I_e:<10.3f} {deviation_percent:<12.2f}")
    
    print("=" * 90)
    print(f"Проанализировано порохов: {analyzed_count}")
    print(f"Исключено порохов: {excluded_count}")
    
    return results, {
        'ballistite_single_channel_avg': ball_tubular_avg,
        'pyroxylin_single_channel_avg': pyro_tubular_avg,
        'pyroxylin_multi_channel_avg': pyro_seven_avg
    }

def print_best_powders(results):
    """Вывод в консоль лучших порохов (отклонение в пределах [-30%;30%])"""
    print("\n" + "=" * 80)
    print("ЛУЧШИЕ ПОРОХА (отклонение в пределах [-30%;30%])")
    print("=" * 80)
    
    category_names = {
        'ballistite_single_channel': 'БАЛЛИСТИТНЫЕ ОДНОКАНАЛЬНЫЕ',
        'pyroxylin_single_channel': 'ПИРОКСИЛИНОВЫЕ ОДНОКАНАЛЬНЫЕ', 
        'pyroxylin_multi_channel': 'ПИРОКСИЛИНОВЫЕ МНОГОКАНАЛЬНЫЕ'
    }
    
    for category, name_ru in category_names.items():
        best_powders = [p for p in results[category] if p['is_best']]
        
        print(f"\n{name_ru}:")
        print("-" * 60)
        if best_powders:
            print(f"{'Название':<20} {'Импульс':<10} {'Отклонение':<12}")
            for powder in best_powders:
                print(f"{powder['name']:<20} {powder['I_e']:<10.3f} {powder['deviation_percent']:<12.2f}%")
            print(f"Всего: {len(best_powders)} порохов")
        else:
            print("Нет порохов, удовлетворяющих критерию")
        print()

def improved_self_annotate_point(ax, dev, imp, name, point_type):
    """
    Улучшенная аннотация одиночной точки со стрелкой
    """
    # ========== НАСТРАИВАЕМЫЕ ПАРАМЕТРЫ ==========
    SINGLE_DISTANCE = 80  # Увеличим расстояние для уменьшения наложений
    SINGLE_ARROW_RAD = 0.25   # Изгиб стрелки
    SINGLE_BBOX_PAD = 0.3     # Уменьшим отступы для компактности
    SINGLE_FONT_SIZE = 6.5    # Немного уменьшим шрифт
    # =============================================
    
    # Определяем сторону для размещения текста
    if dev >= 0:
        ha = 'left'
        xytext = (SINGLE_DISTANCE, 0)
    else:
        ha = 'right'
        xytext = (-SINGLE_DISTANCE, 0)
    
    # Определяем цвет фона в зависимости от типа точки
    if point_type == 'best':
        bbox_color = 'gold'
        bbox_edge = 'darkorange'
        font_weight = 'bold'
    else:
        bbox_color = 'white'
        bbox_edge = 'gray'
        font_weight = 'normal'
    
    ax.annotate(name, (dev, imp), 
               xytext=xytext,
               textcoords='offset points',
               fontsize=SINGLE_FONT_SIZE, 
               fontweight=font_weight,
               ha=ha, 
               va='center',
               bbox=dict(boxstyle='round,pad=' + str(SINGLE_BBOX_PAD),
                       facecolor=bbox_color, 
                       alpha=0.95, 
                       edgecolor=bbox_edge,
                       linewidth=0.8),
               arrowprops=dict(arrowstyle='->', 
                            color='black', 
                            alpha=0.7,
                            linewidth=0.8,
                            connectionstyle=f"arc3,rad={SINGLE_ARROW_RAD}"),
               zorder=10)

def improved_self_annotate_point(ax, dev, imp, name, point_type):
    """
    Улучшенная аннотация одиночной точки со стрелкой
    Лучшие пороха - сверху, обычные - снизу
    """
    # ========== НАСТРАИВАЕМЫЕ ПАРАМЕТРЫ ==========
    SINGLE_DISTANCE = 80      # Расстояние от точки до текста
    SINGLE_ARROW_RAD = 0.25   # Изгиб стрелки
    SINGLE_BBOX_PAD = 0.3     # Отступы вокруг текста
    SINGLE_FONT_SIZE = 6.5    # Размер шрифта
    # =============================================
    
    # ОПРЕДЕЛЯЕМ ПОЛОЖЕНИЕ ПО ТИПУ ПОРОХА
    if point_type == 'best':
        # Лучшие пороха - СВЕРХУ
        va = 'bottom'
        xytext = (0, SINGLE_DISTANCE)  # Смещение вверх
    else:
        # Обычные пороха - СНИЗУ
        va = 'top'
        xytext = (0, -SINGLE_DISTANCE)  # Смещение вниз
    
    # Определяем горизонтальное выравнивание по отклонению
    if dev >= 0:
        ha = 'left'
        xytext = (xytext[0] + SINGLE_DISTANCE * 0.3, xytext[1])  # Немного правее
    else:
        ha = 'right'
        xytext = (xytext[0] - SINGLE_DISTANCE * 0.3, xytext[1])  # Немного левее
    
    # Определяем цвет фона в зависимости от типа точки
    if point_type == 'best':
        bbox_color = 'gold'
        bbox_edge = 'darkorange'
        font_weight = 'bold'
    else:
        bbox_color = 'white'
        bbox_edge = 'gray'
        font_weight = 'normal'
    
    ax.annotate(name, (dev, imp), 
               xytext=xytext,
               textcoords='offset points',
               fontsize=SINGLE_FONT_SIZE, 
               fontweight=font_weight,
               ha=ha, 
               va=va,
               bbox=dict(boxstyle='round,pad=' + str(SINGLE_BBOX_PAD),
                       facecolor=bbox_color, 
                       alpha=0.95, 
                       edgecolor=bbox_edge,
                       linewidth=0.8),
               arrowprops=dict(arrowstyle='->', 
                            color='black', 
                            alpha=0.7,
                            linewidth=0.8,
                            connectionstyle=f"arc3,rad={SINGLE_ARROW_RAD}"),
               zorder=10)

def improved_fan_annotate_points(ax, points):
    """
    Улучшенная аннотация группы близких точек веером
    Лучшие пороха - сверху, обычные - снизу
    """
    # ========== НАСТРАИВАЕМЫЕ ПАРАМЕТРЫ ==========
    FAN_BASE_DISTANCE = 80      # Базовое расстояние от центра
    FAN_DISTANCE_STEP = 10       # Приращение расстояния
    FAN_BASE_ANGLE_RANGE =200   # Диапазон углов распределения
    FAN_ANGLE_PER_POINT = 20    # Угол на точку
    FAN_ARROW_RAD = 0.25          # Изгиб стрелки
    FAN_BBOX_PAD = 0.25           # Отступы вокруг текста
    FAN_FONT_SIZE = 5.8           # Размер шрифта
    # =============================================
    
    # Вычисляем центр группы
    center_dev = np.mean([p[0] for p in points])
    center_imp = np.mean([p[1] for p in points])
    
    # РАЗДЕЛЯЕМ ТОЧКИ ПО ТИПАМ
    best_points = [p for p in points if p[3] == 'best']
    regular_points = [p for p in points if p[3] == 'regular']
    
    # Аннотируем лучшие пороха СВЕРХУ
    if best_points:
        # Сортируем лучшие пороха по импульсу
        best_points.sort(key=lambda x: x[1])
        
        # Определяем базовый угол для лучших порохов (СВЕРХУ)
        if center_dev >= 0:
            base_angle_best = -45  # Верхний правый сектор
        else:
            base_angle_best = 125  # Верхний левый сектор
        
        # Распределяем лучшие точки по дуге
        n_best = len(best_points)
        angle_range_best = min(FAN_BASE_ANGLE_RANGE * 0.7, n_best * FAN_ANGLE_PER_POINT)
        angles_best = np.linspace(-angle_range_best/2, angle_range_best/2, n_best)
        
        for i, (dev, imp, name, point_type) in enumerate(best_points):
            # Вычисляем угол и расстояние для лучших порохов
            angle = base_angle_best + angles_best[i]
            distance = FAN_BASE_DISTANCE + i * FAN_DISTANCE_STEP
            
            # Конвертируем полярные координаты в декартовы
            rad_angle = np.radians(angle)
            xytext = (distance * np.cos(rad_angle), distance * np.sin(rad_angle))
            
            # Определяем выравнивание текста
            if center_dev >= 0:
                ha = 'left'
            else:
                ha = 'right'
            
            ax.annotate(name, (dev, imp), 
                       xytext=xytext,
                       textcoords='offset points',
                       fontsize=FAN_FONT_SIZE,
                       fontweight='bold',
                       ha=ha, 
                       va='center',
                       bbox=dict(boxstyle='round,pad=' + str(FAN_BBOX_PAD),
                               facecolor='gold', 
                               alpha=0.95, 
                               edgecolor='darkorange',
                               linewidth=0.6),
                       arrowprops=dict(arrowstyle='->', 
                                    color='black', 
                                    alpha=0.6,
                                    linewidth=0.6,
                                    connectionstyle=f"arc3,rad={FAN_ARROW_RAD}"),
                       zorder=10)
    
    # Аннотируем обычные пороха СНИЗУ
    if regular_points:
        # Сортируем обычные пороха по импульсу
        regular_points.sort(key=lambda x: x[1])
        
        # Определяем базовый угол для обычных порохов (СНИЗУ)
        if center_dev >= 0:
            base_angle_regular = 150 # Нижний правый сектор
        else:
            base_angle_regular = -75  # Нижний левый сектор
        
        # Распределяем обычные точки по дуге
        n_regular = len(regular_points)
        angle_range_regular = min(FAN_BASE_ANGLE_RANGE * 0.7, n_regular * FAN_ANGLE_PER_POINT)
        angles_regular = np.linspace(-angle_range_regular/2, angle_range_regular/2, n_regular)
        
        for i, (dev, imp, name, point_type) in enumerate(regular_points):
            # Вычисляем угол и расстояние для обычных порохов
            angle = base_angle_regular + angles_regular[i]
            distance = FAN_BASE_DISTANCE + i * FAN_DISTANCE_STEP
            
            # Конвертируем полярные координаты в декартовы
            rad_angle = np.radians(angle)
            xytext = (distance * np.cos(rad_angle), distance * np.sin(rad_angle))
            
            # Определяем выравнивание текста
            if center_dev >= 0:
                ha = 'left'
            else:
                ha = 'right'
            
            ax.annotate(name, (dev, imp), 
                       xytext=xytext,
                       textcoords='offset points',
                       fontsize=FAN_FONT_SIZE,
                       fontweight='normal',
                       ha=ha, 
                       va='center',
                       bbox=dict(boxstyle='round,pad=' + str(FAN_BBOX_PAD),
                               facecolor='white', 
                               alpha=0.95, 
                               edgecolor='gray',
                               linewidth=0.6),
                       arrowprops=dict(arrowstyle='->', 
                                    color='black', 
                                    alpha=0.6,
                                    linewidth=0.6,
                                    connectionstyle=f"arc3,rad={FAN_ARROW_RAD}"),
                       zorder=10)

def improved_plot_results(results, avg_values):
    """
    Улучшенное построение графиков с разделением по типам порохов
    """
    # ========== УЛУЧШЕННЫЕ ПАРАМЕТРЫ ГРУППИРОВКИ ==========
    GROUP_THRESHOLD_DEV = 2.3  # Порог по отклонению
    GROUP_THRESHOLD_IMP = 0.02    # Порог по импульсу
    MARGIN_FACTOR_X = 0.5    # Увеличим отступы по X для размещения подписей
    MARGIN_FACTOR_Y = 0.6       # Увеличим отступы по Y для размещения подписей
    # ======================================================
    
    categories = [
        ('ballistite_single_channel', '', 'red'),
        ('pyroxylin_single_channel', '', 'green'),
        ('pyroxylin_multi_channel', '', 'orange')
    ]
    
    # Создаем отдельные фигуры для каждого графика
    for category, title, default_color in categories:
        data = results[category]
        
        if not data:
            print(f"Нет данных для категории: {title}")
            continue
        
        # Создаем новую фигуру для половины листа А4 (8.27 x 5.845 дюймов)
        fig, ax = plt.subplots(figsize=(8.27, 5.845))
        
        # Разделяем данные на обычные и лучшие
        regular_data = [d for d in data if not d['is_best']]
        best_data = [d for d in data if d['is_best']]
        
        # Строим график для обычных порохов
        if regular_data:
            deviations_regular = [d['deviation_percent'] for d in regular_data]
            impulses_regular = [d['I_e'] for d in regular_data]
            names_regular = [d['name'] for d in regular_data]
            
            scatter_regular = ax.scatter(deviations_regular, impulses_regular, 
                                       c=default_color, alpha=0.8, s=60, 
                                       edgecolors='black', linewidth=0.6, zorder=5,
                                       label='Обычные пороха')
        
        # Строим график для лучших порохов (золотистым цветом)
        if best_data:
            deviations_best = [d['deviation_percent'] for d in best_data]
            impulses_best = [d['I_e'] for d in best_data]
            names_best = [d['name'] for d in best_data]
            
            scatter_best = ax.scatter(deviations_best, impulses_best, 
                                    c='gold', alpha=1.0, s=80, 
                                    edgecolors='darkorange', linewidth=1.0, zorder=6,
                                    label='Лучшие пороха (±30%)')
        
        # Добавляем среднюю линию
        avg_key = f"{category}_avg"
        if avg_key in avg_values:
            avg_line = avg_values[avg_key]
            ax.axhline(y=avg_line, color='darkred', linestyle='--', alpha=0.8, 
                      linewidth=2.0, label=f'Среднее: {avg_line:.3f} МПа·с', zorder=3)
        
        # Настройки графика (НЕ МЕНЯЕМ положение статистики и легенды)
        ax.set_xlabel('Отклонение от среднего (%)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Импульс пороха (МПа·с)', fontsize=10, fontweight='bold')
        ax.set_title(f'{title}', fontsize=12, fontweight='bold', pad=10)
        
        # Улучшенная сетка
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.3)
        ax.set_axisbelow(True)
        
        # Настраиваем шкалы для лучшего отображения (с увеличенными отступами)
        all_deviations = [d['deviation_percent'] for d in data]
        all_impulses = [d['I_e'] for d in data]
        
        x_range = max(all_deviations) - min(all_deviations)
        y_range = max(all_impulses) - min(all_impulses)
        
        x_margin = x_range * MARGIN_FACTOR_X if x_range > 0 else 15
        y_margin = y_range * MARGIN_FACTOR_Y if y_range > 0 else 0.15
        
        ax.set_xlim(min(all_deviations) - x_margin, max(all_deviations) + x_margin)
        ax.set_ylim(min(all_impulses) - y_margin, max(all_impulses) + y_margin)
        
        # Добавляем вертикальные линии для диапазона ±30%
        ax.axvline(x=-30, color='gold', linestyle=':', alpha=0.7, linewidth=1.5, zorder=2)
        ax.axvline(x=30, color='gold', linestyle=':', alpha=0.7, linewidth=1.5, zorder=2)
        
        # УЛУЧШЕННОЕ ДОБАВЛЕНИЕ АННОТАЦИЙ С РАЗДЕЛЕНИЕМ ПО ТИПАМ
        all_data = []
        if regular_data:
            all_data.extend(zip(deviations_regular, impulses_regular, names_regular, ['regular']*len(regular_data)))
        if best_data:
            all_data.extend(zip(deviations_best, impulses_best, names_best, ['best']*len(best_data)))
        
        # Сортируем данные по отклонению для последовательного расположения
        all_data.sort(key=lambda x: x[0])
        
        # УЛУЧШЕННЫЙ АЛГОРИТМ ГРУППИРОВКИ
        point_groups = {}
        used_points = set()
        
        # Проходим по всем точкам и создаем группы
        for i, (dev, imp, name, point_type) in enumerate(all_data):
            if (dev, imp, name) in used_points:
                continue
                
            # Ищем ближайшие точки для группировки
            current_group = [(dev, imp, name, point_type)]
            used_points.add((dev, imp, name))
            
            for j, (dev2, imp2, name2, point_type2) in enumerate(all_data[i+1:], i+1):
                if (dev2, imp2, name2) in used_points:
                    continue
                    
                # Проверяем расстояние до текущей группы
                distance_to_group = min(
                    abs(dev2 - d) + abs(imp2 - i) * 10  # Импульс имеет больший вес
                    for d, i, _, _ in current_group
                )
                
                if distance_to_group < GROUP_THRESHOLD_DEV * 2:
                    current_group.append((dev2, imp2, name2, point_type2))
                    used_points.add((dev2, imp2, name2))
            
            # Сохраняем группу
            if len(current_group) > 0:
                group_center_dev = np.mean([p[0] for p in current_group])
                group_center_imp = np.mean([p[1] for p in current_group])
                point_groups[(group_center_dev, group_center_imp)] = current_group
        
        # Добавляем аннотации для каждой группы
        for group_key, points in point_groups.items():
            if len(points) == 1:
                # Одиночная точка
                dev, imp, name, point_type = points[0]
                improved_self_annotate_point(ax, dev, imp, name, point_type)
            else:
                # Группа точек
                improved_fan_annotate_points(ax, points)
        
        # СТАТИСТИКА - БЕЗ ИЗМЕНЕНИЙ (левый верхний угол)
        stats_text = (f'Всего: {len(data)}\n'
                     f'Лучших: {len(best_data)}\n'
                     f'Ср. откл.: {np.mean(all_deviations):.1f}%\n'
                     f'Макс. откл.: {max(abs(d) for d in all_deviations):.1f}%\n'
                     f'Импульсы: {min(all_impulses):.3f}-{max(all_impulses):.3f}')
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', 
                bbox=dict(boxstyle='round,pad=0.5', 
                        facecolor='lightblue', 
                        alpha=0.8,
                        edgecolor='navy'))
        
        # ЛЕГЕНДА - БЕЗ ИЗМЕНЕНИЙ (правый верхний угол)
        ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
        
        # Улучшаем тики на осях
        ax.tick_params(axis='both', which='major', labelsize=8)
        
        # Добавляем вертикальную линию в ноль для ориентира
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5, linewidth=0.8)
        
        # Оптимизируем layout для половины А4
        plt.tight_layout(pad=2.0)
        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.12, right=0.93)
        
        # Сохраняем график в файл
        filename = f"improved_powder_analysis_{category}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Улучшенный график сохранен как: {filename}")
        
        plt.show()
        
        # Выводим статистику в консоль
        print(f"\nСтатистика для {title}:")
        print(f"  Количество порохов: {len(data)}")
        print(f"  Лучших порохов: {len(best_data)}")
        print(f"  Среднее отклонение: {np.mean(all_deviations):.2f}%")
        print(f"  Минимальное отклонение: {min(all_deviations):.2f}%")
        print(f"  Максимальное отклонение: {max(all_deviations):.2f}%")
        print(f"  Средний импульс: {np.mean(all_impulses):.3f} МПа·с")
        print(f"  Диапазон импульсов: {min(all_impulses):.3f}-{max(all_impulses):.3f} МПа·с")

# В основной функции замените вызов plot_results на improved_plot_results
def main():
    print("АНАЛИЗ БАЗЫ ДАННЫХ ПОРОХОВ")
    print("=" * 50)
    


    # Анализируем базу данных порохов
    results, avg_values = analyze_powders_database()
    
            # ВЫВОДИМ НАЗВАНИЯ ЛУЧШИХ ПОРОХОВ ЧЕРЕЗ ЗАПЯТУЮ
    print_approved_powders_names(results)
    
    # Выводим лучшие пороха в консоль
    print_best_powders(results)
    
    # Строим УЛУЧШЕННЫЕ графики
    improved_plot_results(results, avg_values)
    
    # СОХРАНЯЕМ ЛУЧШИЕ ПОРОХА В ФАЙЛ
    approved_powders = save_approved_powders(results)
    

    
    return results, avg_values

def save_approved_powders(results):
    """Сохранение лучших порохов в файл approved_baza.py"""
    
    # Собираем все лучшие пороха
    approved_powders = {}
    
    for category in results.keys():
        for powder in results[category]:
            if powder['is_best']:
                # Находим оригинальные данные пороха из базы
                powder_name = powder['name']
                if powder_name in powders_db:
                    approved_powders[powder_name] = powders_db[powder_name]
    
    # Формируем содержимое файла
    file_content = "# База данных лучших порохов (отклонение в пределах ±30%)\n"
    file_content += "approved_powders_db = {\n"
    
    for name, data in approved_powders.items():
        file_content += f"    '{name}': {{\n"
        file_content += f"        'I_e': {data['I_e']},\n"
        file_content += f"        'z_e': {data['z_e']},\n"
        file_content += f"        'delta': {data['delta']},\n"
        file_content += f"        'rho': {data['rho']},\n"
        file_content += f"        'f': {data['f']},\n"
        file_content += f"        'k': {data['k']},\n"
        file_content += f"        'alpha_k': {data['alpha_k']},\n"
        file_content += f"        'T_1': {data['T_1']},\n"
        file_content += f"        'kappa_1': {data['kappa_1']},\n"
        file_content += f"        'lambda_1': {data['lambda_1']},\n"
        file_content += f"        'mu_1': {data['mu_1']},\n"
        file_content += f"        'T_2': {data['T_2']},\n"
        file_content += f"        'kappa_2': {data['kappa_2']},\n"
        file_content += f"        'lambda_2': {data['lambda_2']},\n"
        file_content += f"        'T_3': {data['T_3']},\n"
        file_content += f"        'kappa_3': {data['kappa_3']},\n"
        file_content += f"        'lambda_3': {data['lambda_3']},\n"
        file_content += f"        'T_4': {data['T_4']},\n"
        file_content += f"        'kappa_4': {data['kappa_4']},\n"
        file_content += f"        'lambda_4': {data['lambda_4']},\n"
        file_content += f"        'T_5': {data['T_5']},\n"
        file_content += f"        'kappa_5': {data['kappa_5']},\n"
        file_content += f"        'lambda_5': {data['lambda_5']},\n"
        file_content += f"        'T_6': {data['T_6']},\n"
        file_content += f"        'kappa_6': {data['kappa_6']},\n"
        file_content += f"        'lambda_6': {data['lambda_6']},\n"
        file_content += f"        'T_7': {data['T_7']},\n"
        file_content += f"        'kappa_7': {data['kappa_7']},\n"
        file_content += f"        'lambda_7': {data['lambda_7']},\n"
        file_content += f"        'T_8': {data['T_8']},\n"
        file_content += f"        'kappa_8': {data['kappa_8']},\n"
        file_content += f"        'lambda_8': {data['lambda_8']},\n"
        file_content += f"        'T_9': {data['T_9']},\n"
        file_content += f"        'kappa_9': {data['kappa_9']},\n"
        file_content += f"        'lambda_9': {data['lambda_9']},\n"
        file_content += f"        'T_10': {data['T_10']},\n"
        file_content += f"        'kappa_10': {data['kappa_10']},\n"
        file_content += f"        'lambda_10': {data['lambda_10']},\n"
        file_content += f"        'T_11': {data['T_11']},\n"
        file_content += f"        'kappa_11': {data['kappa_11']},\n"
        file_content += f"        'lambda_11': {data['lambda_11']},\n"
        file_content += f"        'T_12': {data['T_12']},\n"
        file_content += f"        'kappa_12': {data['kappa_12']},\n"
        file_content += f"        'lambda_12': {data['lambda_12']},\n"
        file_content += f"        'T_13': {data['T_13']},\n"
        file_content += f"        'kappa_13': {data['kappa_13']},\n"
        file_content += f"        'lambda_13': {data['lambda_13']},\n"
        file_content += f"        'T_14': {data['T_14']},\n"
        file_content += f"        'kappa_14': {data['kappa_14']},\n"
        file_content += f"        'lambda_14': {data['lambda_14']},\n"
        file_content += f"        'T_15': {data['T_15']},\n"
        file_content += f"        'kappa_15': {data['kappa_15']},\n"
        file_content += f"        'lambda_15': {data['lambda_15']},\n"
        file_content += f"        'T_16': {data['T_16']},\n"
        file_content += f"        'kappa_16': {data['kappa_16']},\n"
        file_content += f"        'lambda_16': {data['lambda_16']},\n"
        file_content += f"        'T_17': {data['T_17']},\n"
        file_content += f"        'kappa_17': {data['kappa_17']},\n"
        file_content += f"        'lambda_17': {data['lambda_17']},\n"
        file_content += f"        'T_18': {data['T_18']},\n"
        file_content += f"        'kappa_18': {data['kappa_18']},\n"
        file_content += f"        'lambda_18': {data['lambda_18']},\n"
        file_content += f"        'T_19': {data['T_19']},\n"
        file_content += f"        'kappa_19': {data['kappa_19']},\n"
        file_content += f"        'lambda_19': {data['lambda_19']},\n"
        file_content += f"        'T_20': {data['T_20']},\n"
        file_content += f"        'kappa_20': {data['kappa_20']},\n"
        file_content += f"        'lambda_20': {data['lambda_20']},\n"
        file_content += f"        'T_21': {data['T_21']},\n"
        file_content += f"        'kappa_21': {data['kappa_21']},\n"
        file_content += f"        'lambda_21': {data['lambda_21']},\n"
        file_content += f"        'T_22': {data['T_22']},\n"
        file_content += f"        'kappa_22': {data['kappa_22']},\n"
        file_content += f"        'lambda_22': {data['lambda_22']},\n"
        file_content += f"        'T_23': {data['T_23']},\n"
        file_content += f"        'kappa_23': {data['kappa_23']},\n"
        file_content += f"        'lambda_23': {data['lambda_23']},\n"
        file_content += f"        'T_24': {data['T_24']},\n"
        file_content += f"        'kappa_24': {data['kappa_24']},\n"
        file_content += f"        'lambda_24': {data['lambda_24']},\n"
        file_content += f"        'T_25': {data['T_25']},\n"
        file_content += f"        'kappa_25': {data['kappa_25']},\n"
        file_content += f"        'lambda_25': {data['lambda_25']},\n"
        file_content += f"        'T_26': {data['T_26']},\n"
        file_content += f"        'kappa_26': {data['kappa_26']},\n"
        file_content += f"        'lambda_26': {data['lambda_26']},\n"
        file_content += f"        'T_27': {data['T_27']},\n"
        file_content += f"        'kappa_27': {data['kappa_27']},\n"
        file_content += f"        'lambda_27': {data['lambda_27']},\n"
        file_content += f"        'T_28': {data['T_28']},\n"
        file_content += f"        'kappa_28': {data['kappa_28']},\n"
        file_content += f"        'lambda_28': {data['lambda_28']},\n"
        file_content += f"        'T_29': {data['T_29']},\n"
        file_content += f"        'kappa_29': {data['kappa_29']},\n"
        file_content += f"        'lambda_29': {data['lambda_29']},\n"
        file_content += f"        'T_30': {data['T_30']},\n"
        file_content += f"        'kappa_30': {data['kappa_30']},\n"
        file_content += f"        'lambda_30': {data['lambda_30']},\n"
        file_content += f"    }},\n"
    
    file_content += "}\n"
    
    # Сохраняем файл
    with open('approved_baza.py', 'w', encoding='utf-8') as f:
        f.write(file_content)
    
    print(f"\nСоздан файл approved_baza.py с {len(approved_powders)} лучшими порохами")
    
    return approved_powders

def print_approved_powders_names(results):
    """Вывод названий всех лучших порохов через запятую"""
    
    approved_names = []
    
    for category in results.keys():
        for powder in results[category]:
            if powder['is_best']:
                approved_names.append(powder['name'])
    
    print("\n" + "=" * 80)
    print("ВСЕ ЛУЧШИЕ ПОРОХА (через запятую):")
    print("=" * 80)
    print(", ".join(approved_names))
    print(f"\nВсего лучших порохов: {len(approved_names)}")


# Запуск программы
if __name__ == "__main__":
    results, avg_values = main()