import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, chi2_contingency, pointbiserialr
import statsmodels.api as sm
import pingouin as pg
import warnings

warnings.filterwarnings('ignore')

# Настройки графиков
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

print("=" * 70)
print("Практическая работа 6: Корреляционный анализ")
print("Датасет: Влияние социальных сетей на отношения и успеваемость")
print("=" * 70)

# 1. ЗАГРУЗКА ДАННЫХ
# Предполагается, что файл 'social_media_addiction.csv' находится в рабочей папке
file_path = 'social_media_addiction.csv'
try:
    df = pd.read_csv(file_path)
    print(f"\nЗагружен файл {file_path}, размер: {df.shape}")
except FileNotFoundError:
    print(f"\nФайл {file_path} не найден.")
    print("Скачайте датасет с Kaggle и поместите его в рабочую папку.")
    print("Ссылка: https://www.kaggle.com/datasets/adilshamim8/social-media-addiction-vs-relationships")
    # Для демонстрации создадим синтетические данные (аналог структуры)
    print("Создаю примерный датасет для демонстрации логики (замените на реальный).")
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        'age': np.random.randint(14, 30, n),
        'gender': np.random.choice(['Male', 'Female'], n),
        'daily_usage_minutes': np.random.randint(30, 480, n),
        'social_media_addiction_score': np.random.uniform(1, 5, n),
        'academic_performance': np.random.uniform(1, 5, n),
        'sleep_hours': np.random.uniform(4, 9, n),
        'physical_activity_hours': np.random.uniform(0, 4, n),
        'relationship_satisfaction': np.random.uniform(1, 5, n),
        'anxiety_level': np.random.uniform(1, 5, n),
        'depression_level': np.random.uniform(1, 5, n)
    })
    # Добавим искусственные связи
    df['daily_usage_minutes'] = df['daily_usage_minutes'] + 50 * (5 - df['academic_performance'])
    df['social_media_addiction_score'] = 5.5 - 0.5 * df['relationship_satisfaction'] + np.random.normal(0, 0.5, n)
    df = df.clip(lower=0)
    print("Синтетические данные созданы. Для реального анализа используйте настоящий CSV.")

# Первичный осмотр
print("\nПервые 5 строк:")
print(df.head())
print("\nИнформация о данных:")
print(df.info())
print("\nОписательная статистика:")
print(df.describe())

# Определим числовые и категориальные столбцы
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"\nЧисловые признаки: {numeric_cols}")
print(f"Категориальные признаки: {categorical_cols}")

# Пропуски – удалим строки с любыми пропусками (при необходимости)
df_clean = df.dropna().copy()
print(f"Строк после удаления пропусков: {len(df_clean)}")

# ==================================================
# ЗАДАНИЕ 1: Корреляционный анализ и проверка значимости
# ==================================================
print("\n" + "=" * 60)
print("ЗАДАНИЕ 1: Корреляционный анализ числовых и категориальных признаков")
print("=" * 60)

# 1.1 Анализ числовых признаков (матрица корреляций, тепловая карта, проверка значимости)
num_data = df_clean[numeric_cols]
if len(numeric_cols) >= 2:
    # Матрица Пирсона
    pearson_corr = num_data.corr(method='pearson')
    # Матрица Спирмена
    spearman_corr = num_data.corr(method='spearman')

    # Визуализация
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', center=0, ax=axes[0],
                fmt='.2f', linewidths=0.5)
    axes[0].set_title('Корреляции Пирсона')
    sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', center=0, ax=axes[1],
                fmt='.2f', linewidths=0.5)
    axes[1].set_title('Корреляции Спирмена')
    plt.tight_layout()
    plt.show()

    # Проверка значимости для каждой пары (на примере наиболее сильных)
    print("\nПроверка значимости корреляций (p-value):")
    threshold = 0.7  # рассматриваем пары с |r| > 0.7
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            r_p, p_p = pearsonr(num_data[numeric_cols[i]], num_data[numeric_cols[j]])
            r_s, p_s = spearmanr(num_data[numeric_cols[i]], num_data[numeric_cols[j]])
            if abs(r_p) > threshold or abs(r_s) > threshold:
                print(
                    f"  {numeric_cols[i]} vs {numeric_cols[j]}: Пирсона r={r_p:.3f}, p={p_p:.4f}; Спирмена r={r_s:.3f}, p={p_s:.4f}")

    # Дополнительно: пары с высокой корреляцией, например daily_usage и academic_performance
    if 'daily_usage_minutes' in numeric_cols and 'academic_performance' in numeric_cols:
        r_p, p_p = pearsonr(df_clean['daily_usage_minutes'], df_clean['academic_performance'])
        r_s, p_s = spearmanr(df_clean['daily_usage_minutes'], df_clean['academic_performance'])
        print(
            f"\nСвязь daily_usage_minutes ↔ academic_performance: Пирсона r={r_p:.3f} (p={p_p:.4f}), Спирмена r={r_s:.3f} (p={p_s:.4f})")
        if p_p < 0.05:
            print("  Коэффициент статистически значим (p<0.05).")
        else:
            print("  Коэффициент статистически незначим.")

# 1.2 Анализ категориальных признаков (таблицы сопряженности, хи-квадрат)
if len(categorical_cols) >= 2:
    print("\nАнализ категориальных признаков:")
    for i in range(len(categorical_cols)):
        for j in range(i + 1, len(categorical_cols)):
            cat1 = categorical_cols[i]
            cat2 = categorical_cols[j]
            contingency = pd.crosstab(df_clean[cat1], df_clean[cat2])
            chi2, p, dof, expected = chi2_contingency(contingency)
            print(f"  {cat1} vs {cat2}: chi2={chi2:.2f}, p-value={p:.4f}")
            if p < 0.05:
                print(f"    Связь статистически значима (p<0.05).")
            else:
                print(f"    Связь не обнаружена.")
    # Визуализация первой пары категорий
    if len(categorical_cols) >= 2:
        sns.heatmap(pd.crosstab(df_clean[categorical_cols[0]], df_clean[categorical_cols[1]]),
                    annot=True, cmap='Blues', fmt='d')
        plt.title(f"Таблица сопряженности: {categorical_cols[0]} vs {categorical_cols[1]}")
        plt.show()

# 1.3 Анализ смешанных пар (числовая + бинарная) с помощью pointbiserialr
# Находим бинарную категориальную переменную (с двумя уникальными значениями)
binary_cat_col = None
for col in categorical_cols:
    if df_clean[col].nunique() == 2:
        binary_cat_col = col
        break

# Если нет бинарной категории, создаём на основе медианы какого-либо числа
if binary_cat_col is None and len(df_clean) > 0:
    numeric_cols_only = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols_only:
        med = df_clean[numeric_cols_only[0]].median()
        df_clean['_binary'] = (df_clean[numeric_cols_only[0]] > med).astype(int)
        binary_cat_col = '_binary'
        print(f"\nСоздана искусственная дихотомическая переменная '{binary_cat_col}' на основе медианы {numeric_cols_only[0]}")
    else:
        binary_cat_col = None

# Если нашли бинарную переменную, преобразуем её в 0/1 и считаем бисериальную корреляцию
if binary_cat_col:
    # Преобразование в 0/1
    if binary_cat_col == '_binary':
        binary_series = df_clean['_binary']
    else:
        # Кодируем два уникальных значения в 1 и 0
        unique_vals = df_clean[binary_cat_col].unique()
        if len(unique_vals) == 2:
            mapping = {unique_vals[0]: 1, unique_vals[1]: 0}
            binary_series = df_clean[binary_cat_col].map(mapping).astype(int)
        else:
            binary_series = None

    if binary_series is not None:
        print(f"\nБисериальная корреляция (числовые vs {binary_cat_col}):")
        # Отбираем только числовые столбцы для сравнения
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        # Убираем саму бинарную переменную, если она попала в список
        if binary_cat_col in numeric_cols:
            numeric_cols.remove(binary_cat_col)
        if '_binary' in numeric_cols:
            numeric_cols.remove('_binary')
        # Для каждого числа считаем корреляцию с бинарной переменной
        for col in numeric_cols:
            # Принудительно приводим к float на случай целых
            x = binary_series.values.astype(float)
            y = df_clean[col].astype(float).values
            try:
                r, p = pointbiserialr(x, y)
                print(f"  {col}: r={r:.3f}, p-value={p:.4f} -> {'статистически значимо' if p < 0.05 else 'не значимо'}")
            except Exception as e:
                print(f"  Ошибка при расчёте для {col}: {e}")
    else:
        print("Не удалось преобразовать бинарную переменную в числовой формат.")
else:
    print("Нет подходящей бинарной переменной для бисериальной корреляции.")

# 1.4 Визуализация распределений для ключевых признаков
key_vars = [v for v in ['daily_usage_minutes', 'academic_performance', 'relationship_satisfaction', 'sleep_hours'] if
            v in numeric_cols]
if key_vars:
    # Парные диаграммы рассеяния
    sns.pairplot(df_clean[key_vars], diag_kind='kde', plot_kws={'alpha': 0.6})
    plt.suptitle('Диаграммы рассеяния ключевых переменных', y=1.02)
    plt.show()

    # Boxplot для связи категориальной и числовой (если есть категории)
    if categorical_cols:
        cat_for_box = categorical_cols[0]
        if cat_for_box in df_clean.columns and len(df_clean[cat_for_box].unique()) <= 5:
            plt.figure()
            sns.boxplot(data=df_clean, x=cat_for_box, y='daily_usage_minutes', palette='Set2')
            plt.title(f'Распределение времени в соцсетях по {cat_for_box}')
            plt.show()

# ==================================================
# ЗАДАНИЕ *: Частные коэффициенты корреляции (исключение влияния других переменных)
# ==================================================
print("\n" + "="*60)
print("ЗАДАНИЕ *: Частные коэффициенты корреляции (исключение влияния других переменных)")
print("="*60)

def partial_corr(data, x, y, controls):
    # Берём только строки без NaN по всем участвующим переменным
    vars_all = [x, y] + controls
    df_clean_partial = data[vars_all].dropna()
    if len(df_clean_partial) < len(controls) + 3:
        return None, None
    # Регрессия x на controls, получаем остатки
    X_ctrl = df_clean_partial[controls].values
    X_ctrl = sm.add_constant(X_ctrl)
    model_x = sm.OLS(df_clean_partial[x], X_ctrl).fit()
    resid_x = model_x.resid
    # Регрессия y на controls
    model_y = sm.OLS(df_clean_partial[y], X_ctrl).fit()
    resid_y = model_y.resid
    # Корреляция между остатками
    r, p = pearsonr(resid_x, resid_y)
    return r, p

# Определим числовые столбцы (без Student_ID)
numeric_for_corr = [c for c in numeric_cols if c != 'Student_ID']

if len(numeric_for_corr) >= 3:
    # Пример 1: связь между временем в соцсетях и конфликтами, контролируя психическое здоровье
    x1 = 'Avg_Daily_Usage_Hours'
    y1 = 'Conflicts_Over_Social_Media'
    control1 = ['Mental_Health_Score']
    r_partial, p_partial = partial_corr(df_clean, x1, y1, control1)
    if r_partial is not None:
        print(f"\nЧастная корреляция между {x1} и {y1} (контроль: {control1}):")
        print(f"  r = {r_partial:.4f}, p-value = {p_partial:.4f}")
        # Обычная корреляция для сравнения
        r_pear, p_pear = pearsonr(df_clean[x1], df_clean[y1])
        print(f"Обычная корреляция Пирсона: r = {r_pear:.4f}, p = {p_pear:.4f}")
    else:
        print(f"\nНедостаточно данных для {x1}, {y1}, {control1}")

    # Пример 2: сон и психическое здоровье с контролем времени в соцсетях
    x2 = 'Sleep_Hours_Per_Night'
    y2 = 'Mental_Health_Score'
    control2 = ['Avg_Daily_Usage_Hours']
    r_partial2, p_partial2 = partial_corr(df_clean, x2, y2, control2)
    if r_partial2 is not None:
        print(f"\nЧастная корреляция между {x2} и {y2} (контроль: {control2}):")
        print(f"  r = {r_partial2:.4f}, p-value = {p_partial2:.4f}")
        r_pear2, p_pear2 = pearsonr(df_clean[x2], df_clean[y2])
        print(f"Обычная корреляция Пирсона: r = {r_pear2:.4f}, p = {p_pear2:.4f}")

    # Пример 3: время в соцсетях и психическое здоровье с контролем по сну и конфликтам
    x3 = 'Avg_Daily_Usage_Hours'
    y3 = 'Mental_Health_Score'
    control3 = ['Sleep_Hours_Per_Night', 'Conflicts_Over_Social_Media']
    r_partial3, p_partial3 = partial_corr(df_clean, x3, y3, control3)
    if r_partial3 is not None:
        print(f"\nЧастная корреляция между {x3} и {y3} (контроль: {control3}):")
        print(f"  r = {r_partial3:.4f}, p-value = {p_partial3:.4f}")
        r_pear3, p_pear3 = pearsonr(df_clean[x3], df_clean[y3])
        print(f"Обычная корреляция Пирсона: r = {r_pear3:.4f}, p = {p_pear3:.4f}")
    else:
        print("\nНедостаточно данных для третьей модели.")
else:
    print("Недостаточно числовых переменных (нужно хотя бы 3) для расчёта частных корреляций.")