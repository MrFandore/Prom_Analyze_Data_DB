import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

print("="*70)
print("ФИНАЛЬНЫЙ ПРОЕКТ – АНАЛИЗ РЫНКА ПОДЕРЖАННЫХ АВТОМОБИЛЕЙ")
print("="*70)

# 1. ЗАГРУЗКА ДАННЫХ
print("\n1. Загрузка данных...")
df = pd.read_csv('listings.csv')
print(f"Размер: {df.shape[0]} строк, {df.shape[1]} столбцов")
print("\nПервые 3 строки:")
print(df.head(3))

# 2. ПРЕДОБРАБОТКА
df['mileage'] = df['mileage'].astype(str).str.replace(' km', '', regex=False).str.replace(' ', '', regex=False)
df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')  # "unknown" станет NaN
# Извлекаем год из first-registration (формат MM-YYYY)
def extract_year(date_str):
    try:
        return pd.to_datetime(date_str, format='%m-%Y').year
    except:
        return np.nan

df['year'] = df['first-registration'].apply(extract_year)

# Удаляем строки с критическими пропусками
initial_len = len(df)
df = df.dropna(subset=['price', 'mileage', 'year'])
print(f"Удалено {initial_len - len(df)} строк с пропущенными значениями")

# Очистка от выбросов (ценовые рамки, пробег, год)
df = df[(df['price'] > 1000) & (df['price'] < 100000)]
df = df[df['mileage'] >= 0]
df = df[df['year'] >= 2000]

print(f"После очистки осталось {len(df)} записей")

if len(df) == 0:
    print("Нет данных после очистки. Проверьте исходный CSV.")
    exit()

# 3. ОПИСАТЕЛЬНАЯ СТАТИСТИКА
print("\nСтатистика по числовым признакам:")
print(df[['price', 'mileage', 'year']].describe())

# 4. ВИЗУАЛИЗАЦИЯ РАСПРЕДЕЛЕНИЙ
fig, axes = plt.subplots(1, 3, figsize=(15,4))
sns.histplot(df['price'], bins=30, kde=True, ax=axes[0], color='steelblue')
axes[0].set_title('Распределение цены (евро)')
sns.histplot(df['mileage'], bins=30, kde=True, ax=axes[1], color='orange')
axes[1].set_title('Распределение пробега (км)')
sns.histplot(df['year'], bins=20, kde=True, ax=axes[2], color='green')
axes[2].set_title('Распределение года выпуска')
plt.tight_layout()
plt.show()

# 5. ЗАВИСИМОСТЬ ЦЕНЫ ОТ ПРОБЕГА И ГОДА
fig, axes = plt.subplots(1, 2, figsize=(12,5))
sns.scatterplot(data=df, x='year', y='price', alpha=0.5, ax=axes[0])
axes[0].set_title('Цена vs год выпуска')
sns.scatterplot(data=df, x='mileage', y='price', alpha=0.5, ax=axes[1])
axes[1].set_title('Цена vs пробег')
plt.tight_layout()
plt.show()

# Корреляции (только если есть вариация)
if df['year'].nunique() > 1 and df['price'].nunique() > 1:
    corr_year = pearsonr(df['year'], df['price'])
    print(f"\nКорреляция цена-год: r={corr_year[0]:.3f}, p={corr_year[1]:.3e}")
if df['mileage'].nunique() > 1:
    corr_mile = pearsonr(df['mileage'], df['price'])
    print(f"Корреляция цена-пробег: r={corr_mile[0]:.3f}, p={corr_mile[1]:.3e}")

# 6. АНАЛИЗ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ
# Топ-10 марок по частоте
top_makes = df['make'].value_counts().head(10)
print("\nТоп-10 самых частых марок:")
print(top_makes)

# Средняя цена по маркам (минимум 5 наблюдений)
make_counts = df['make'].value_counts()
valid_makes = make_counts[make_counts >= 5].index
avg_price_make = df[df['make'].isin(valid_makes)].groupby('make')['price'].mean().sort_values(ascending=False).head(10)
print("\nТоп-10 самых дорогих марок (средняя цена, мин. 5 авто):")
print(avg_price_make)

if len(avg_price_make) > 0:
    plt.figure(figsize=(12,6))
    sns.barplot(y=avg_price_make.index, x=avg_price_make.values, hue=avg_price_make.index, palette='viridis', legend=False)
    plt.title('Средняя цена по маркам')
    plt.xlabel('Цена (евро)')
    plt.tight_layout()
    plt.show()

# Влияние типа топлива (если есть осмысленные значения)
if 'fuel-type' in df.columns:
    fuel_map = {'2': 'Бензин', 'b': 'Дизель', 'e': 'Электро', 'h': 'Гибрид'}
    df['fuel_desc'] = df['fuel-type'].map(fuel_map).fillna(df['fuel-type'])
    plt.figure(figsize=(8,5))
    sns.boxplot(data=df, x='fuel_desc', y='price')
    plt.title('Цена в зависимости от типа топлива')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 7. ВОЗРАСТ АВТОМОБИЛЯ
current_year = 2026
df['age'] = current_year - df['year']
print(f"\nСредний возраст автомобиля: {df['age'].mean():.1f} лет")
sns.histplot(df['age'], bins=25, kde=True, color='purple')
plt.title('Распределение возраста автомобилей')
plt.show()

# 8. ОБЩИЕ ВЫВОДЫ
print("\n" + "="*70)
print("ВЫВОДЫ ПО РЕЗУЛЬТАТАМ АНАЛИЗА")
print("="*70)
print(f"1. Всего проанализировано {len(df)} автомобилей, представленных на площадке.")
print("2. Цены варьируются от {:.0f} до {:.0f} €, средняя цена = {:.0f} €.".format(df['price'].min(), df['price'].max(), df['price'].mean()))
print("3. Пробег: от {:.0f} до {:.0f} км, средний = {:.0f} км.".format(df['mileage'].min(), df['mileage'].max(), df['mileage'].mean()))
print("4. Годы выпуска: от {} до {} (средний = {})".format(df['year'].min(), df['year'].max(), round(df['year'].mean())))
print("5. Наблюдается отрицательная корреляция между ценой и пробегом (r ≈ {:.2f}) – чем больше пробег, тем дешевле авто.".format(corr_mile[0] if 'corr_mile' in locals() else 0))
print("6. Положительная корреляция цена-год (r ≈ {:.2f}) – более новые машины дороже.".format(corr_year[0] if 'corr_year' in locals() else 0))
print("7. Самые дорогие марки среди представленных:", ", ".join(avg_price_make.head(3).index.tolist()))
print("8. Средний возраст автомобиля – около {:.1f} лет, что говорит о преобладании относительно свежих машин.".format(df['age'].mean()))