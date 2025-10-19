import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def read_results_file(filename):
    """Чтение данных из файла results.txt"""
    try:
        # Читаем файл как CSV с разделителем запятой
        df = pd.read_csv(filename)
        return df
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return None

def plot_results(data):
    """Построение графика"""
    # Создаем фигуру и оси
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Определяем маркеры для разного количества порохов (добавлен маркер для 4 порохов)
    markers = {
        1: 'o',  # круг для одного пороха
        2: 's',  # квадрат для двух порохов
        3: '*',  # звезда для трех порохов
        4: 'D',  # ромб для четырех порохов
    }
    
    marker_sizes = {
        1: 100,   # размер для круга
        2: 80,    # размер для квадрата
        3: 120,   # размер для звезды
        4: 100    # размер для ромба
    }
    
    # Получаем уникальные типы смесей для цветовой шкалы
    unique_types = data['type'].unique()
    n_types = len(unique_types)
    
    # Создаем цветовую карту для типов
    colors = plt.cm.Set3(np.linspace(0, 1, n_types))
    type_color_map = {type_name: color for type_name, color in zip(unique_types, colors)}
    
    # Создаем scatter plot для каждого типа num_powders
    scatter_plots = []
    labels = []
    
    for num_powders in sorted(data['num_powders'].unique()):
        if num_powders in markers:
            # Фильтруем данные по количеству порохов
            filtered_data = data[data['num_powders'] == num_powders]
            
            # Создаем scatter plot
            for i, row in filtered_data.iterrows():
                scatter = ax.scatter(
                    row['p_max_mpa'], 
                    row['x_pm_m'], 
                    c=[type_color_map[row['type']]],
                    marker=markers[num_powders],
                    s=marker_sizes[num_powders],
                    alpha=0.7,
                    edgecolors='black',
                    linewidth=0.8
                )
            
            # Сохраняем один экземпляр для легенды
            scatter_plots.append(scatter)
            labels.append(f'{num_powders} порох(а)')
    
    # Создаем легенду для типов смесей
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=type_color_map[type_name], label=type_name) 
                      for type_name in unique_types]
    
    ax.legend(handles=legend_elements, title='Типы смесей:', 
              loc='upper left', bbox_to_anchor=(1, 1), framealpha=0.9)
    
    # Настройка осей и заголовка
    ax.set_xlabel('Максимальное давление, p_max_mpa (МПа)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Длина ствола, x_pm_m (м)', fontsize=12, fontweight='bold')
    ax.set_title('Зависимость длины ствола от максимального давления\n'
                'Цвет: тип смеси | Маркеры: количество порохов', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Добавляем легенду для маркеров
    ax.legend(scatter_plots, labels, title='Количество порохов:', 
              loc='upper right', framealpha=0.9)
    
    # Добавляем обратно легенду для типов (первая легенда перезаписалась)
    ax.legend(handles=legend_elements, title='Типы смесей:', 
              loc='upper left', bbox_to_anchor=(1, 1), framealpha=0.9)
    
    # Сетка для лучшей читаемости
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Улучшаем внешний вид
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Автоматическая настройка layout
    plt.tight_layout()
    
    # Показываем график
    plt.show()

def plot_results_improved(data):
    """Улучшенная версия графика с двумя легендами"""
    # Создаем фигуру и оси
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Определяем маркеры для разного количества порохов (добавлен маркер для 4 порохов)
    markers = {
        1: 'o',  # круг для одного пороха
        2: 's',  # квадрат для двух порохов
        3: '*',  # звезда для трех порохов
        4: 'D',  # ромб для четырех порохов
    }
    
    marker_sizes = {
        1: 100,   # размер для круга
        2: 80,    # размер для квадрата
        3: 120,   # размер для звезды
        4: 100    # размер для ромба
    }
    
    # Получаем уникальные типы смесей для цветовой шкалы
    unique_types = data['type'].unique()
    n_types = len(unique_types)
    
    # Создаем цветовую карту для типов
    colors = plt.cm.Set3(np.linspace(0, 1, n_types))
    type_color_map = {type_name: color for type_name, color in zip(unique_types, colors)}
    
    # Создаем отдельные scatter plots для легенды маркеров
    marker_legend_elements = []
    
    for num_powders in sorted(data['num_powders'].unique()):
        if num_powders in markers:
            # Фильтруем данные по количеству порохов
            filtered_data = data[data['num_powders'] == num_powders]
            
            # Группируем по типам для лучшего отображения
            for type_name in filtered_data['type'].unique():
                type_data = filtered_data[filtered_data['type'] == type_name]
                
                scatter = ax.scatter(
                    type_data['p_max_mpa'], 
                    type_data['x_pm_m'], 
                    c=[type_color_map[type_name]],
                    marker=markers[num_powders],
                    s=marker_sizes[num_powders],
                    alpha=0.7,
                    edgecolors='black',
                    linewidth=0.8,
                    label=f'{num_powders} порох(а)' if type_name == filtered_data['type'].iloc[0] else ""
                )
            
            # Добавляем элемент для легенды маркеров
            marker_legend_elements.append(
                plt.Line2D([0], [0], marker=markers[num_powders], color='gray', 
                          markersize=10, label=f'{num_powders} порох(а)', 
                          linestyle='None', markeredgecolor='black')
            )
    
    # Создаем легенду для типов смесей
    from matplotlib.patches import Patch
    type_legend_elements = [Patch(facecolor=type_color_map[type_name], label=type_name) 
                          for type_name in unique_types]
    
    # Добавляем первую легенду (типы смесей)
    type_legend = ax.legend(handles=type_legend_elements, title='Типы смесей:', 
                           loc='upper left', bbox_to_anchor=(1, 1), framealpha=0.9)
    ax.add_artist(type_legend)
    
    # Добавляем вторую легенду (маркеры)
    ax.legend(handles=marker_legend_elements, title='Количество порохов:', 
              loc='upper left', bbox_to_anchor=(1, 0.7), framealpha=0.9)
    
    # Настройка осей и заголовка
    ax.set_xlabel('Максимальное давление, p_max_mpa (МПа)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Длина ствола, x_pm_m (м)', fontsize=12, fontweight='bold')
    ax.set_title('Зависимость длины ствола от максимального давления\n'
                'Цвет: тип смеси | Маркеры: количество порохов', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Сетка для лучшей читаемости
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Улучшаем внешний вид
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Автоматическая настройка layout
    plt.tight_layout()
    
    # Показываем график
    plt.show()

def print_statistics(data):
    """Вывод статистики по данным"""
    print("Статистика данных:")
    print(f"Всего записей: {len(data)}")
    print(f"Типы смесей: {list(data['type'].unique())}")
    print(f"Количество порохов в смесях: {sorted(data['num_powders'].unique())}")
    print(f"Диапазон давления: {data['p_max_mpa'].min():.1f} - {data['p_max_mpa'].max():.1f} МПа")
    print(f"Диапазон длины ствола: {data['x_pm_m'].min():.3f} - {data['x_pm_m'].max():.3f} м")
    print("\n")

def main():
    """Основная функция"""
    try:
        # Чтение данных из файла
        data = read_results_file('results.txt')
        
        if data is None or data.empty:
            print("Файл results.txt пуст или содержит некорректные данные")
            return
        
        print(f"Прочитано {len(data)} записей")
        print_statistics(data)
        
        # Проверяем наличие необходимых столбцов
        required_columns = ['p_max_mpa', 'x_pm_m', 'type', 'num_powders']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            print(f"Отсутствуют необходимые столбцы: {missing_columns}")
            print(f"Доступные столбцы: {list(data.columns)}")
            return
        
        # Построение графика (используем улучшенную версию)
        plot_results_improved(data)
        
    except FileNotFoundError:
        print("Файл results.txt не найден")
        print("Убедитесь, что файл находится в той же директории, что и программа")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

# Дополнительная функция для графика с аннотациями
def plot_detailed_results(data):
    """Построение графика с аннотациями названий смесей"""
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Обновленные маркеры с поддержкой 4 порохов
    markers = {1: 'o', 2: 's', 3: '*', 4: 'D'}
    
    # Получаем уникальные типы смесей
    unique_types = data['type'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_types)))
    type_color_map = {type_name: color for type_name, color in zip(unique_types, colors)}
    
    for num_powders in sorted(data['num_powders'].unique()):
        if num_powders in markers:
            filtered_data = data[data['num_powders'] == num_powders]
            
            for type_name in filtered_data['type'].unique():
                type_data = filtered_data[filtered_data['type'] == type_name]
                
                ax.scatter(
                    type_data['p_max_mpa'], 
                    type_data['x_pm_m'], 
                    c=[type_color_map[type_name]],
                    marker=markers[num_powders],
                    s=100,
                    alpha=0.7,
                    edgecolors='black',
                    linewidth=0.8
                )
    
    # Добавляем аннотации с названиями смесей
    for i, row in data.iterrows():
        ax.annotate(row['name'], 
                   (row['p_max_mpa'], row['x_pm_m']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.7)
    
    # Легенды
    from matplotlib.patches import Patch
    type_legend_elements = [Patch(facecolor=type_color_map[type_name], label=type_name) 
                          for type_name in unique_types]
    
    marker_legend_elements = [
        plt.Line2D([0], [0], marker=markers[num], color='gray', markersize=10, 
                  label=f'{num} порох(а)', linestyle='None', markeredgecolor='black')
        for num in markers.keys() if num in data['num_powders'].unique()
    ]
    
    ax.legend(handles=type_legend_elements + marker_legend_elements, 
              title='Условные обозначения', loc='best', framealpha=0.9)
    
    ax.set_xlabel('Максимальное давление, p_max_mpa (МПа)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Длина ствола, x_pm_m (м)', fontsize=12, fontweight='bold')
    ax.set_title('Зависимость длины ствола от максимального давления с названиями смесей', 
                fontsize=14, fontweight='bold')
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
    
    # Раскомментируйте для графика с аннотациями названий
    # data = read_results_file('results.txt')
    # if data is not None and not data.empty:
    #     plot_detailed_results(data)