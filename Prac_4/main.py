import pandas as pd
from scipy.stats import chi2_contingency

# 1. Загрузка данных
df = pd.read_csv('Titanic-Dataset.csv')
print("Размер датасета:", df.shape)
print(df.head())

# 2. Проверка гипотезы: пол и выживание
# Создаём таблицу сопряженности
contingency_sex_survived = pd.crosstab(df['Sex'], df['Survived'])
print("\nТаблица сопряженности (Sex x Survived):")
print(contingency_sex_survived)

chi2, p, dof, expected = chi2_contingency(contingency_sex_survived)
print(f"\nХи-квадрат: {chi2:.4f}")
print(f"p-value: {p:.6f}")
print(f"Степени свободы: {dof}")
print("Ожидаемые частоты:")
print(expected)

alpha = 0.05
if p < alpha:
    print("Вывод: Отвергаем нулевую гипотезу о независимости пола и выживания.")
    print("Пол влияет на выживаемость.")
else:
    print("Вывод: Нет оснований отвергать нулевую гипотезу. Пол не связан с выживаемостью.")

# 3. Создание категориальной переменной "ребёнок/взрослый"
# Возраст ребёнка – до 18 лет (в некоторых источниках 15, но возьмём 18)
df['Age'] = df['Age'].fillna(df['Age'].median())  # заполняем пропуски медианой
df['Child'] = df['Age'].apply(lambda x: 'Child' if x < 18 else 'Adult')
print("\nРаспределение по категории Child:")
print(df['Child'].value_counts())

# Таблица сопряженности Child x Survived
contingency_child_survived = pd.crosstab(df['Child'], df['Survived'])
print("\nТаблица сопряженности (Child x Survived):")
print(contingency_child_survived)

chi2_child, p_child, dof_child, expected_child = chi2_contingency(contingency_child_survived)
print(f"\nХи-квадрат: {chi2_child:.4f}")
print(f"p-value: {p_child:.6f}")

if p_child < alpha:
    print("Вывод: Отвергаем нулевую гипотезу о независимости возраста и выживания.")
    print("Возраст (ребёнок/взрослый) влияет на выживаемость.")
else:
    print("Вывод: Нет оснований отвергать нулевую гипотезу. Возраст не связан с выживаемостью.")