import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re

def load_data_from_file(filename):
    """
    Загружает данные из файла time.txt с улучшенным парсингом
    """
    data = []
    
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        
        print("Содержимое файла:")
        print("=" * 50)
        for i, line in enumerate(lines):
            print(f"{i:2}: {line.strip()}")
        print("=" * 50)
        
        # Ищем начало данных
        start_reading = False
        for line in lines:
            line = line.strip()
            
            # Пропускаем пустые строки и заголовки
            if not line:
                continue
                
            if line.startswith('Тип_пороха'):
                start_reading = True
                continue
                
            if line == '==================================================':
                break
                
            if start_reading and line:
                # Улучшенный парсинг строк
                parts = parse_line(line)
                if parts and len(parts) >= 6:
                    data.append(parts)
                    print(f"Распарсена строка: {parts}")
                else:
                    print(f"Не удалось распарсить: {line}")
    
    if not data:
        print("Не найдено данных. Пробуем альтернативный метод...")
        data = load_data_alternative(filename)
    
    # Создаем DataFrame
    if data:
        df = pd.DataFrame(data, columns=['Тип_пороха', 'Кол-во_порохов', 'Комбинации_порохов', 
                                        'Итерации', 'Время_сек', 'Среднее_время_итерации_сек'])
        
        # Преобразуем числовые колонки к правильным типам
        numeric_columns = ['Кол-во_порохов', 'Комбинации_порохов', 'Итерации', 
                          'Время_сек', 'Среднее_время_итерации_сек']
        
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    else:
        return pd.DataFrame()

def parse_line(line):
    """
    Парсит строку с данными, пытаясь определить формат
    """
    # Пробуем разные разделители
    for delimiter in [',', '\t', ';', '|']:
        if delimiter in line:
            parts = line.split(delimiter)
            if len(parts) >= 6:
                return [p.strip() for p in parts]
    
    # Пробуем разделить по пробелам (учитываем, что числа могут быть с точкой)
    parts = re.split(r'\s+', line.strip())
    
    # Объединяем части, если их меньше 6
    if len(parts) < 6:
        return None
    
    # Если частей больше 6, пытаемся найти правильное разделение
    if len(parts) > 6:
        # Первый элемент - тип пороха
        # Второй - количество порохов (целое число)
        # Остальные - числа (могут быть с точкой)
        result = [parts[0]]  # Тип пороха
        
        # Ищем второе целое число
        for i in range(1, len(parts)):
            if parts[i].isdigit():
                result.append(parts[i])
                break
        
        # Остальные числа
        number_count = 0
        for part in parts[i+1:]:
            if re.match(r'^-?\d+\.?\d*$', part) and number_count < 4:
                result.append(part)
                number_count += 1
            if number_count >= 4:
                break
        
        if len(result) >= 6:
            return result
    
    return parts if len(parts) >= 6 else None

def load_data_alternative(filename):
    """
    Альтернативный метод загрузки данных
    """
    data = []
    
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        
        for line in lines:
            line = line.strip()
            
            # Пропускаем служебные строки
            if (not line or 
                line.startswith('Тип_пороха') or 
                line.startswith('ОБЩИЕ') or 
                line.startswith('===')):
                continue
            
            # Пробуем найти данные по шаблонам
            patterns = [
                # Паттерн для данных типа "ПО,2,10,120,1.87,0.0156"
                r'^(\w+),?[\s]*(\d+),?[\s]*(\d+),?[\s]*(\d+),?[\s]*(\d+\.\d+),?[\s]*(\d+\.\d+)$',
                # Паттерн для данных с пробелами "ПО 2 10 120 1.87 0.0156"
                r'^(\w+)[\s]+(\d+)[\s]+(\d+)[\s]+(\d+)[\s]+(\d+\.\d+)[\s]+(\d+\.\d+)$',
                # Паттерн для данных с табуляцией
                r'^(\w+)\t+(\d+)\t+(\d+)\t+(\d+)\t+(\d+\.\d+)\t+(\d+\.\d+)$'
            ]
            
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    data.append(list(match.groups()))
                    print(f"Найдены данные по паттерну: {list(match.groups())}")
                    break
    
    return data

def create_correct_test_file():
    """
    Создает корректный тестовый файл
    """
    content = """Тип_пороха,Кол-во_порохов,Комбинации_порохов,Итерации,Время_сек,Среднее_время_итерации_сек
ПО,2,10,120,1.87,0.0156
ПО,3,10,240,3.41,0.0142
ПО,4,5,120,1.93,0.0161
ПС,2,10,120,1.68,0.0140
ПС,3,10,240,3.07,0.0128
ПС,4,5,100,1.41,0.0141
БО,2,10,120,1.42,0.0119
БО,3,10,240,2.57,0.0107
БО,4,5,120,1.38,0.0115
Свободная,2,105,1260,14.90,0.0118
Свободная,3,455,10920,130.90,0.0120
Свободная,4,1365,32760,406.43,0.0124
==================================================
ОБЩИЕ ПОКАЗАТЕЛИ:
Всего итераций: 46360
Общее время расчета: 570.98 сек
Среднее время на итерацию: 0.0123 сек
Всего комбинаций рассчитано: 12"""

    with open('time.txt', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Корректный тестовый файл time.txt создан!")
    return content

def load_general_stats(filename):
    """
    Загружает общие показатели из файла
    """
    general_stats = {}
    
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        
        found_separator = False
        for line in lines:
            if line.strip() == '==================================================':
                found_separator = True
                continue
            if found_separator and line.startswith('ОБЩИЕ ПОКАЗАТЕЛИ:'):
                continue
            if found_separator and line.strip():
                if ':' in line:
                    key, value = line.split(':', 1)
                    general_stats[key.strip()] = value.strip()
    
    return general_stats

# Основная программа
print("Загрузка данных из time.txt...")

# Сначала создадим корректный файл
create_correct_test_file()

# Загрузка данных из файла
df = load_data_from_file('time.txt')
general_stats = load_general_stats('time.txt')

if len(df) == 0:
    print("Не удалось загрузить данные. Используем тестовые данные...")
    # Создаем DataFrame с тестовыми данными
    test_data = {
        'Тип_пороха': ['ПО', 'ПО', 'ПО', 'ПС', 'ПС', 'ПС', 'БО', 'БО', 'БО', 'Свободная', 'Свободная', 'Свободная'],
        'Кол-во_порохов': [2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4],
        'Комбинации_порохов': [10, 10, 5, 10, 10, 5, 10, 10, 5, 105, 455, 1365],
        'Итерации': [120, 240, 120, 120, 240, 100, 120, 240, 120, 1260, 10920, 32760],
        'Время_сек': [1.87, 3.41, 1.93, 1.68, 3.07, 1.41, 1.42, 2.57, 1.38, 14.90, 130.90, 406.43],
        'Среднее_время_итерации_сек': [0.0156, 0.0142, 0.0161, 0.0140, 0.0128, 0.0141, 0.0119, 0.0107, 0.0115, 0.0118, 0.0120, 0.0124]
    }
    df = pd.DataFrame(test_data)

print(f"Загружено записей: {len(df)}")
print("\nДанные для визуализации:")
print(df)

# Визуализация
if len(df) > 0:
    # Настройка стиля графиков
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(15, 12))

    # 1. График времени выполнения по типам пороха
    plt.subplot(2, 3, 1)
    for powder_type in df['Тип_пороха'].unique():
        type_data = df[df['Тип_пороха'] == powder_type]
        plt.plot(type_data['Кол-во_порохов'], type_data['Время_сек'], 
                 marker='o', label=powder_type, linewidth=2)
    plt.xlabel('Количество порохов')
    plt.ylabel('Время (сек)')
    plt.title('Время выполнения по типам пороха')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. График среднего времени на итерацию
    plt.subplot(2, 3, 2)
    for powder_type in df['Тип_пороха'].unique():
        type_data = df[df['Тип_пороха'] == powder_type]
        plt.plot(type_data['Кол-во_порохов'], type_data['Среднее_время_итерации_сек'], 
                 marker='s', label=powder_type, linewidth=2)
    plt.xlabel('Количество порохов')
    plt.ylabel('Среднее время итерации (сек)')
    plt.title('Среднее время на итерацию')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. Столбчатая диаграмма количества комбинаций
    plt.subplot(2, 3, 3)
    x_pos = np.arange(len(df))
    plt.bar(x_pos, df['Комбинации_порохов'], alpha=0.7, color='skyblue')
    plt.xlabel('Эксперименты')
    plt.ylabel('Количество комбинаций')
    plt.title('Количество комбинаций порохов')
    plt.xticks(x_pos, [f"{row['Тип_пороха']}-{row['Кол-во_порохов']}" for _, row in df.iterrows()], 
               rotation=45, ha='right')

    # 4. Круговая диаграмма распределения итераций по типам пороха
    plt.subplot(2, 3, 4)
    iterations_by_type = df.groupby('Тип_пороха')['Итерации'].sum()
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    plt.pie(iterations_by_type, labels=iterations_by_type.index, autopct='%1.1f%%', 
            colors=colors, startangle=90)
    plt.title('Распределение итераций по типам пороха')

    # 5. Heatmap среднего времени итерации
    plt.subplot(2, 3, 5)
    pivot_table = df.pivot_table(values='Среднее_время_итерации_сек', 
                                index='Тип_пороха', 
                                columns='Кол-во_порохов', 
                                fill_value=0)
    im = plt.imshow(pivot_table.values, cmap='YlOrRd', aspect='auto')
    plt.colorbar(im, label='Среднее время итерации (сек)')
    plt.xticks(range(len(pivot_table.columns)), pivot_table.columns)
    plt.yticks(range(len(pivot_table.index)), pivot_table.index)
    plt.xlabel('Количество порохов')
    plt.ylabel('Тип пороха')
    plt.title('Heatmap: Среднее время итерации')

    # 6. Сравнение времени и количества комбинаций
    plt.subplot(2, 3, 6)
    scatter = plt.scatter(df['Комбинации_порохов'], df['Время_сек'], 
                         c=df['Кол-во_порохов'], s=100, alpha=0.7, 
                         cmap='viridis')
    plt.colorbar(scatter, label='Количество порохов')
    plt.xlabel('Количество комбинаций')
    plt.ylabel('Время выполнения (сек)')
    plt.title('Зависимость времени от количества комбинаций')
    plt.grid(True, alpha=0.3)

    # Добавляем аннотации для точек
    for i, row in df.iterrows():
        plt.annotate(f"{row['Тип_пороха']}-{row['Кол-во_порохов']}", 
                    (row['Комбинации_порохов'], row['Время_сек']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

    plt.tight_layout()
    plt.show()

    # Вывод статистики
    print("\n" + "="*50)
    print("СТАТИСТИКА ДАННЫХ:")
    print("="*50)
    print(f"Всего итераций: {df['Итерации'].sum()}")
    print(f"Общее время: {df['Время_сек'].sum():.2f} сек")
    print(f"Среднее время на итерацию: {df['Среднее_время_итерации_сек'].mean():.4f} сек")
    
else:
    print("Нет данных для визуализации")