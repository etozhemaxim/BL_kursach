import matplotlib.pyplot as plt

# Данные
powders = [2, 3, 4]
combinations = [1854360, 185436000, 18358164000]
labels = ['1,854,360', '185,436,000', '18.36×10⁹']

# Создаем гистограмму
plt.figure(figsize=(8, 5))
bars = plt.bar(powders, combinations, color='steelblue', edgecolor='navy')

# Настройки
plt.xlabel('Число порохов в смеси')
plt.ylabel('Число комбинаций ')
plt.xticks(powders)
plt.yscale('log')

# Добавляем подписи
for bar, label in zip(bars, labels):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), label,
             ha='center', va='bottom', fontweight='bold')

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()