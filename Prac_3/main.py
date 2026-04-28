import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Загрузка данных
df = sns.load_dataset("diamonds")
print("Датасет загружен. Размер:", df.shape)
print("\nПервые 5 строк:")
print(df.head())

#Общая информация о данных
print("\nИнформация о данных:")
print(df.info())

#Проверка на пропуски
print("\nПропуски по столбцам:")
print(df.isnull().sum())

#Удаление дубликатов
duplicates = df.duplicated().sum()
print(f"\nКоличество дубликатов: {duplicates}")
if duplicates > 0:
    df = df.drop_duplicates()
    print("Дубликаты удалены. Новый размер:", df.shape)

#Описательные статистики для числовых столбцов
print("\nОписательная статистика (числовые):")
print(df.describe())

#Описательные статистики для категориальных столбцов
print("\nОписательная статистика (категориальные):")
print(df.describe(include=['object', 'category']))

#Распределение числовых признаков (гистограммы)
num_cols = ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']
for col in num_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], bins=50, kde=True)
    plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.show()

#Гистограмма цены с ограничением по оси X для лучшего обзора
plt.figure(figsize=(8, 4))
sns.histplot(df['price'], bins=100, kde=True)
plt.xlim(0, 20000)
plt.title("Distribution of price (up to 20000)")
plt.show()

#Категориальные переменные: частоты (countplot)
cat_cols = ['cut', 'color', 'clarity']
for col in cat_cols:
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(data=df, x=col, order=df[col].value_counts().index)
    ax.set_title(f'Count of {col}')
    plt.setp(ax.get_xticklabels(), rotation=45)
    plt.tight_layout()
    plt.show()

#Ящики с усами для цены по категориям
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, col in enumerate(cat_cols):
    sns.boxplot(data=df, x=col, y='price', ax=axes[i])
    axes[i].set_title(f'Price by {col}')
    plt.setp(axes[i].get_xticklabels(), rotation=45)
plt.tight_layout()
plt.show()

#Корреляция числовых признаков (без дублирующих размеров x, y, z)
num_for_corr = ['carat', 'depth', 'table', 'price']
corr = df[num_for_corr].corr()
plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation matrix")
plt.tight_layout()
plt.show()

#Диаграмма рассеяния: цена vs вес
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='carat', y='price', alpha=0.3)
plt.title("Price vs Carat")
plt.tight_layout()
plt.show()

#Группировка 1: средняя цена по комбинации огранка-цвет-прозрачность
grouped = df.groupby(['cut', 'color', 'clarity'])['price'].mean().reset_index()
print("\nСредняя цена по группам (первые 10 самых дорогих):")
print(grouped.sort_values('price', ascending=False).head(10))

#Группировка 2: средние значения цены, карат и глубины по качеству огранки
print("\nСредние значения по качеству огранки:")
print(df.groupby('cut')[['price', 'carat', 'depth']].mean().sort_values('price', ascending=False))

#Тепловая карта средних цен по цвету и прозрачности
pivot = df.pivot_table(values='price', index='color', columns='clarity', aggfunc='mean')
plt.figure(figsize=(10, 6))
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='viridis')
plt.title("Средняя цена по цвету и прозрачности")
plt.tight_layout()
plt.show()

#Анализ выбросов (IQR)
print("\nАнализ выбросов по методу IQR:")
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
    print(f"{col}: {len(outliers)} выбросов ({(len(outliers)/len(df))*100:.2f}%)")

print("\nАнализ завершён.")