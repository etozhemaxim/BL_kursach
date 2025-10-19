import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_and_process_data(filename):
    """Загрузка и обработка данных"""
    df = pd.read_csv(filename)
    
    # Преобразование типов данных
    numeric_columns = ['p_max_mpa', 'v_pm_mps', 'x_pm_m', 'omega_sum_kg', 'W_0_m3', 'delta_kg_m3']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Фильтрация только ПС и БО
    df = df[df['type'].isin(['ПС', 'БО'])]
    
    return df

def create_pressure_vs_length_plot(df):
    """Создание графика зависимости давления от длины ствола"""
    
    # Создаем фигуру
    plt.figure(figsize=(12, 8))
    
    # Определяем маркеры для разного числа порохов
    markers = {2: 'o', 3: 's', 4: '^', 5: 'D'}  # круг, квадрат, треугольник, ромб
    colors = {'ПС': 'red', 'БО': 'blue'}
    
    # Рисуем точки для каждого типа и количества порохов
    for powder_type in ['ПС', 'БО']:
        type_data = df[df['type'] == powder_type]
        
        for num_powders in sorted(type_data['num_powders'].unique()):
            mask = type_data['num_powders'] == num_powders
            subset = type_data[mask]
            
            plt.scatter(subset['x_pm_m'], subset['p_max_mpa'],
                       marker=markers.get(num_powders, 'o'),
                       c=colors[powder_type],
                       s=80, alpha=0.7,
                       label=f'{powder_type}, {num_powders} порохов')
    
    plt.xlabel('Длина ствола, м', fontsize=12)
    plt.ylabel('Максимальное давление, МПа', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Улучшаем легенду
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()

def find_best_solutions(df):
    """Нахождение лучших решений по длине ствола"""
    
    print("ЛУЧШИЕ РЕШЕНИЯ ПО ДЛИНЕ СТВОЛА:")
    print("=" * 50)
    
    # Сортируем по длине ствола
    df_sorted = df.sort_values('x_pm_m')
    
    # Находим лучшие решения для каждой длины ствола (минимальное давление при достижении скорости ~950 м/с)
    best_solutions = []
    
    for x_length in sorted(df_sorted['x_pm_m'].unique()):
        length_data = df_sorted[df_sorted['x_pm_m'] == x_length]
        
        # Ищем решение с минимальным давлением при скорости близкой к 950 м/с
        target_speed_data = length_data[length_data['v_pm_mps'] >= 940]
        
        if len(target_speed_data) > 0:
            best_solution = target_speed_data.loc[target_speed_data['p_max_mpa'].idxmin()]
        else:
            # Если нет решений со скоростью >= 940 м/с, берем лучшее по эффективности
            best_solution = length_data.loc[length_data['efficiency_v_p'].idxmax()]
        
        best_solutions.append(best_solution)
    
    # Создаем DataFrame с лучшими решениями
    best_df = pd.DataFrame(best_solutions)
    
    # Выводим таблицу лучших решений
    print("\nТоп-10 лучших решений:")
    results = best_df[['name', 'type', 'num_powders', 'x_pm_m', 'p_max_mpa', 'v_pm_mps', 'omega_sum_kg']].head(10)
    
    for _, row in results.iterrows():
        print(f"{row['name']:15} | {row['type']:2} | {row['num_powders']:1} порох | "
              f"Длина: {row['x_pm_m']:5.3f} м | "
              f"Давление: {row['p_max_mpa']:6.1f} МПа | "
              f"Скорость: {row['v_pm_mps']:6.1f} м/с | "
              f"Масса: {row['omega_sum_kg']:4.2f} кг")
    
    return best_df

# Основная программа
if __name__ == "__main__":
    # Загрузка данных
    df = load_and_process_data('results.txt')
    
    # Добавляем эффективность для анализа
    df['efficiency_v_p'] = df['v_pm_mps'] / df['p_max_mpa']
    
    # Создаем график
    create_pressure_vs_length_plot(df)
    
    # Находим и выводим лучшие решения
    best_solutions = find_best_solutions(df)