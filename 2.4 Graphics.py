import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Настройка стиля
plt.style.use('default')
plt.rcParams['font.family'] = 'DejaVu Sans'

def load_and_process_data(filename):
    """Загрузка и обработка данных"""
    df = pd.read_csv(filename)
    
    # Преобразование типов данных
    numeric_columns = ['p_max_mpa', 'v_pm_mps', 'x_pm_m', 'omega_sum_kg', 'W_0_m3', 'delta_kg_m3']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Фильтрация только типа "Свободная" и только 2, 3, 4 пороха
    df = df[(df['type'] == 'Свободная') & (df['num_powders'].isin([2, 3, 4]))]
    
    return df

def create_pressure_vs_length_plot(df):
    """Создание графика зависимости давления от длины ствола"""
    
    if len(df) == 0:
        print("Нет данных для типа 'Свободная' с 2, 3, 4 порохами")
        return
    
    # Создаем фигуру
    plt.figure(figsize=(12, 8))
    
    # Дискретная цветовая схема
    colors = {2: 'red', 3: 'blue', 4: 'green'}
    markers = {2: 'o', 3: 's', 4: '^'}
    
    # Рисуем точки для каждого количества порохов
    for num_powders in [2, 3, 4]:
        mask = df['num_powders'] == num_powders
        subset = df[mask]
        
        if len(subset) > 0:
            plt.scatter(subset['x_pm_m'], subset['p_max_mpa'],
                       c=colors[num_powders],
                       marker=markers[num_powders],
                       s=80,
                       alpha=0.7,
                       edgecolors='black',
                       linewidth=0.8,
                       label=f'{num_powders} пороха')
    
    plt.xlabel('Длина ствола, м', fontsize=12, fontweight='bold')
    plt.ylabel('Максимальное давление, МПа', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def detailed_analysis(df):
    """Детальный статистический анализ"""
    
    if len(df) == 0:
        return
    
    print("=" * 60)
    print("ДЕТАЛЬНЫЙ АНАЛИЗ: ТИП 'СВОБОДНАЯ' (2-4 ПОРОХА)")
    print("=" * 60)
    
    # Основная статистика
    print(f"\nОбщее количество смесей: {len(df)}")
    
    # Статистика по количеству порохов
    powder_count_stats = df['num_powders'].value_counts().sort_index()
    print(f"\nРаспределение по числу порохов:")
    for num, count in powder_count_stats.items():
        print(f"  {num} пороха: {count} смесей")
    
    # Статистика по группам
    print(f"\nСтатистика по группам:")
    stats_by_powders = df.groupby('num_powders').agg({
        'p_max_mpa': ['count', 'mean', 'std', 'min', 'max'],
        'x_pm_m': ['mean', 'std', 'min', 'max'],
        'v_pm_mps': ['mean', 'std'],
        'omega_sum_kg': ['mean', 'std']
    }).round(3)
    
    print(stats_by_powders)
    
    # Анализ эффективности
    df['efficiency'] = df['v_pm_mps'] / df['p_max_mpa']
    
    print(f"\nАнализ эффективности (скорость/давление):")
    efficiency_stats = df.groupby('num_powders')['efficiency'].agg(['mean', 'std', 'min', 'max']).round(4)
    print(efficiency_stats)

def create_comparison_boxplots(df):
    """Создание boxplot для сравнения распределений"""
    
    if len(df) == 0:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Boxplot для давлений
    pressure_data = [df[df['num_powders'] == num]['p_max_mpa'] for num in [2, 3, 4]]
    ax1.boxplot(pressure_data, labels=['2 пороха', '3 пороха', '4 пороха'])
    ax1.set_ylabel('Максимальное давление (МПа)', fontweight='bold')
    ax1.set_title('Распределение давлений по числу порохов', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Boxplot для длин стволов
    length_data = [df[df['num_powders'] == num]['x_pm_m'] for num in [2, 3, 4]]
    ax2.boxplot(length_data, labels=['2 пороха', '3 пороха', '4 пороха'])
    ax2.set_ylabel('Длина ствола (м)', fontweight='bold')
    ax2.set_title('Распределение длин стволов по числу порохов', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def find_optimal_solutions(df):
    """Поиск оптимальных решений"""
    
    if len(df) == 0:
        return
    
    print(f"\n" + "=" * 60)
    print("ОПТИМАЛЬНЫЕ РЕШЕНИЯ")
    print("=" * 60)
    
    # Лучшие решения по эффективности для каждого числа порохов
    for num_powders in [2, 3, 4]:
        mask = df['num_powders'] == num_powders
        subset = df[mask]
        
        if len(subset) > 0:
            best_efficiency = subset.nlargest(3, 'efficiency')
            print(f"\nТоп-3 по эффективности ({num_powders} пороха):")
            for _, row in best_efficiency.iterrows():
                print(f"  {row['name']:30} | Длина: {row['x_pm_m']:5.3f} м | "
                      f"Давление: {row['p_max_mpa']:6.1f} МПа | "
                      f"Эффективность: {row['efficiency']:5.3f}")
import pandas as pd

def load_and_process_data(filename):
    """Загрузка и обработка данных"""
    df = pd.read_csv(filename)
    
    # Преобразование типов данных
    numeric_columns = ['p_max_mpa', 'v_pm_mps', 'x_pm_m', 'omega_sum_kg', 'W_0_m3', 'delta_kg_m3']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Фильтрация только типа "Свободная"
    df = df[df['type'] == 'Свободная']
    
    return df

def find_top_solutions_by_length(df):
    """Поиск 10 лучших решений по длине ствола"""
    
    if len(df) == 0:
        print("Нет данных для типа 'Свободная'")
        return
    
    # Сортируем по длине ствола (чем короче - тем лучше)
    df_sorted = df.sort_values('x_pm_m')
    
    # Берем топ-10 решений
    top_10 = df_sorted.head(10)
    
    print("=" * 80)
    print("ТОП-10 ЛУЧШИХ РЕШЕНИЙ (ТИП: СВОБОДНАЯ)")
    print("Критерий: минимальная длина ствола")
    print("=" * 80)
    
    print(f"{'Марка смеси':<35} {'Длина ствола, м':<15} {'Давление, МПа':<15} {'Число порохов':<12}")
    print("-" * 80)
    
    for i, (index, row) in enumerate(top_10.iterrows(), 1):
        print(f"{i:2}. {row['name']:<32} {row['x_pm_m']:<15.3f} {row['p_max_mpa']:<15.1f} {row['num_powders']:<12}")
    
    # Дополнительная статистика
    print("\n" + "=" * 80)
    print("СТАТИСТИКА ТОП-10 РЕШЕНИЙ:")
    print(f"Диапазон длин стволов: {top_10['x_pm_m'].min():.3f} - {top_10['x_pm_m'].max():.3f} м")
    print(f"Диапазон давлений: {top_10['p_max_mpa'].min():.1f} - {top_10['p_max_mpa'].max():.1f} МПа")
    
    # Распределение по числу порохов
    powder_dist = top_10['num_powders'].value_counts().sort_index()
    print("Распределение по числу порохов:")
    for num, count in powder_dist.items():
        print(f"  {num} пороха: {count} решений")

def find_top_solutions_by_length_with_speed_filter(df, min_speed=940):
    """Поиск лучших решений по длине ствола с фильтром по скорости"""
    
    if len(df) == 0:
        print("Нет данных для типа 'Свободная'")
        return
    
    # Фильтруем по минимальной скорости
    df_filtered = df[df['v_pm_mps'] >= min_speed]
    
    if len(df_filtered) == 0:
        print(f"Нет решений со скоростью ≥ {min_speed} м/с")
        return
    
    # Сортируем по длине ствола
    df_sorted = df_filtered.sort_values('x_pm_m')
    
    # Берем топ-10 решений
    top_10 = df_sorted.head(10)
    
    print("\n" + "=" * 80)
    print(f"ТОП-10 РЕШЕНИЙ СО СКОРОСТЬЮ ≥ {min_speed} М/С")
    print("Критерий: минимальная длина ствола")
    print("=" + 80)
    
    print(f"{'Марка смеси':<35} {'Длина ствола, м':<15} {'Давление, МПа':<15} {'Скорость, м/с':<15}")
    print("-" * 80)
    
    for i, (index, row) in enumerate(top_10.iterrows(), 1):
        print(f"{i:2}. {row['name']:<32} {row['x_pm_m']:<15.3f} {row['p_max_mpa']:<15.1f} {row['v_pm_mps']:<15.1f}")


# Основная программа
if __name__ == "__main__":
    # Загрузка данных
    df = load_and_process_data('results.txt')
    
    if len(df) > 0:
        # Вывод общей информации
        print(f"Найдено решений типа 'Свободная': {len(df)}")
        print(f"Диапазон длин стволов: {df['x_pm_m'].min():.3f} - {df['x_pm_m'].max():.3f} м")
        print(f"Диапазон давлений: {df['p_max_mpa'].min():.1f} - {df['p_max_mpa'].max():.1f} МПа")
        print()
        
        # Топ-10 по длине ствола
        find_top_solutions_by_length(df)
        
        # Топ-10 с фильтром по скорости (опционально)
        find_top_solutions_by_length_with_speed_filter(df, min_speed=940)
        
    else:
        print("Нет данных типа 'Свободная' в файле results.txt")