import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

df = pd.read_csv('animal_data_dirty.csv', sep=';')
print("=== Исходные данные ===")
print(f"Размер: {df.shape}")
print("Первые 5 строк:")
print(df.head())

print("\n=== Описательные статистики (до очистки) ===")
print(df.describe(include='all'))

print("\n=== Типы данных и пропуски ===")
df.info()

print("\n=== Количество пропусков по столбцам ===")
print(df.isnull().sum())

duplicates_before = df.duplicated().sum()
print(f"\n=== Количество дубликатов: {duplicates_before}")

if df.shape[0] > 0:
    try:
        msno.matrix(df)
        plt.title("Матрица пропусков (исходные данные)")
        plt.show()
    except Exception as e:
        print(f"Не удалось построить матрицу пропусков: {e}")

num_cols = df.select_dtypes(include=np.number).columns.tolist()
print(f"\nЧисловые столбцы: {num_cols}")

if len(num_cols) == 0:
    print("Внимание: нет числовых столбцов. Попробуем преобразовать столбцы с числами вручную...")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    print(f"После преобразования числовые столбцы: {num_cols}")

for col in num_cols:
    if df[col].notna().sum() > 0 and df[col].nunique() > 1:
        plt.figure()
        sns.histplot(df[col], kde=True)
        plt.title(f"Распределение {col} (до очистки)")
        plt.show()
    else:
        print(f"Столбец {col} пропущен для гистограммы (мало данных или одно значение)")

for col in num_cols:
    if df[col].notna().sum() > 0 and df[col].nunique() > 1:
        try:
            plt.figure()
            sns.boxplot(y=df[col])
            plt.title(f"Ящик с усами: {col} (до очистки)")
            plt.show()
        except Exception as e:
            print(f"Не удалось построить boxplot для {col}: {e}")
    else:
        print(f"Столбец {col} пропущен для boxplot (мало данных или одно значение)")

for col in num_cols:
    median_val = df[col].median()
    if not np.isnan(median_val):
        df[col] = df[col].fillna(median_val)
    else:
        df[col] = df[col].fillna(0)

cat_cols = df.select_dtypes(include=['object', 'string']).columns
for col in cat_cols:
    mode_val = df[col].mode()
    if not mode_val.empty:
        df[col] = df[col].fillna(mode_val[0])
    else:
        df[col] = df[col].fillna('unknown')
print("\nПропусков после заполнения:", df.isnull().sum().sum())

#Удаление дубликатов
df = df.drop_duplicates(keep='first')
print(f"Размер после удаления дубликатов: {df.shape}")

mask = pd.Series(True, index=df.index)
for col in num_cols:
    if df[col].notna().sum() > 0 and df[col].nunique() > 1:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        if IQR > 0:
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            mask &= (df[col] >= lower) & (df[col] <= upper)

removed_outliers = (~mask).sum()
df = df[mask]
print(f"Удалено выбросов: {removed_outliers} строк")
print(f"Размер после удаления выбросов: {df.shape}")

if df.shape[0] == 0:
    print("\nВНИМАНИЕ: после удаления выбросов не осталось данных. Возможно, границы IQR слишком строгие.")

for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

print("\n=== Типы данных после очистки ===")
df.info()
print("\n=== Поиск противоречий ===")

# Пример 1: возраст > 30 и вес < 1 (нереалистично)
if 'age' in df.columns and 'weight' in df.columns:
    # Преобразуем в числовые, если ещё не
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
    contradict1 = df[(df['age'] > 30) & (df['weight'] < 1)]
    print(f"Противоречий (возраст>30 и вес<1): {len(contradict1)}")
    if len(contradict1) > 0:
        print(contradict1[['age', 'weight']].head())

# Пример 2: вид 'cat', а вес > 50 кг
if 'species' in df.columns and 'weight' in df.columns:
    # Приводим к строке на всякий случай
    df['species'] = df['species'].astype(str)
    contradict2 = df[(df['species'] == 'cat') & (df['weight'] > 50)]
    print(f"Противоречий (cat с весом>50 кг): {len(contradict2)}")
    if len(contradict2) > 0:
        print(contradict2[['species', 'weight']].head())

# Пример 3: отрицательный возраст
if 'age' in df.columns:
    negative_age = df[df['age'] < 0]
    print(f"Отрицательный возраст: {len(negative_age)}")
    if len(negative_age) > 0:
        print(negative_age[['age']].head())

print("\n=== Описательные статистики ПОСЛЕ очистки ===")
print(df.describe(include='all'))

print(f"\nОкончательный размер очищенного датасета: {df.shape}")
if df.shape[0] > 0:
    for col in num_cols:
        if col in df.columns and df[col].notna().sum() > 0 and df[col].nunique() > 1:
            plt.figure()
            sns.histplot(df[col], kde=True)
            plt.title(f"Распределение {col} (после очистки)")
            plt.show()

    for col in num_cols:
        if col in df.columns and df[col].notna().sum() > 0 and df[col].nunique() > 1:
            try:
                plt.figure()
                sns.boxplot(y=df[col])
                plt.title(f"Ящик с усами: {col} (после очистки)")
                plt.show()
            except Exception as e:
                print(f"Не удалось построить boxplot для {col} после очистки: {e}")


df.to_csv('animal_data_cleaned.csv', index=False)