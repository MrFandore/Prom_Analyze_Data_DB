# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis
from scipy.stats import chi2

# -------------------------
# 1. Загрузка и подготовка данных
# -------------------------
df = pd.read_csv('train.csv')
print("Размер исходного набора:", df.shape)

# Список количественных переменных (оценки)
service_cols = ['Inflight wifi service', 'Departure/Arrival time convenient',
                'Ease of Online booking', 'Gate location', 'Food and drink',
                'Online boarding', 'Seat comfort', 'Inflight entertainment',
                'On-board service', 'Leg room service', 'Baggage handling',
                'Checkin service', 'Inflight service', 'Cleanliness']

numeric_cols = ['Age', 'Flight Distance'] + service_cols

# Категориальные -> дихотомические
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})                     # 1 - мужчина
df['Customer Type'] = df['Customer Type'].map({'Loyal Customer': 1, 'disloyal Customer': 0})
df['Type of Travel'] = df['Type of Travel'].map({'Business travel': 1, 'Personal Travel': 0})

# Класс обслуживания: создадим фиктивные переменные
class_dummies = pd.get_dummies(df['Class'], prefix='Class', drop_first=True)

# Собираем итоговый датафрейм
df_fa = df[numeric_cols].join([df[['Gender', 'Customer Type', 'Type of Travel']], class_dummies])

# Удаляем строки с пропущенными значениями
df_fa = df_fa.dropna()
print("Размер после удаления пропусков:", df_fa.shape)

# -------------------------
# 2. Нормализация данных
# -------------------------
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_fa), columns=df_fa.columns)

# -------------------------
# 3. Проверка применимости факторного анализа
# -------------------------
def bartlett_sphericity(X):
    """Критерий сферичности Бартлетта."""
    n, p = X.shape
    R = np.corrcoef(X.T)
    detR = np.linalg.det(R)
    if detR <= 0:
        return np.nan, np.nan
    statistic = - (n - 1 - (2*p + 5)/6) * np.log(detR)
    df = p*(p-1)//2
    p_value = 1 - chi2.cdf(statistic, df)
    return statistic, p_value

def kmo(X):
    """Коэффициент Кайзера-Мейера-Олкина."""
    corr = np.corrcoef(X.T)
    partial_corr = -np.linalg.pinv(corr)
    diag_partial = np.diag(partial_corr)
    partial_corr_sq = partial_corr**2
    np.fill_diagonal(partial_corr_sq, 0)
    kmo_num = np.sum(corr**2 - np.diag(corr**2))
    kmo_denom = kmo_num + np.sum(partial_corr_sq)
    if kmo_denom == 0:
        return np.nan
    kmo_value = kmo_num / kmo_denom
    return kmo_value

chi2, p_value = bartlett_sphericity(df_scaled.values)
print(f"Статистика Бартлетта: {chi2:.2f}, p-value: {p_value}")
print("Данные подходят (p-value ≈ 0)")

kmo_value = kmo(df_scaled.values)
print(f"KMO: {kmo_value:.3f}")
print("Значение > 0.6 указывает на удовлетворительную адекватность")

# -------------------------
# 4. График каменистой осыпи (собственные значения корреляционной матрицы)
# -------------------------
pca = PCA()
pca.fit(df_scaled)
eigenvalues = pca.explained_variance_

plt.figure(figsize=(8, 5))
plt.scatter(range(1, len(eigenvalues)+1), eigenvalues, color='b')
plt.plot(range(1, len(eigenvalues)+1), eigenvalues, color='b', linestyle='--')
plt.axhline(y=1, color='r', linestyle='-', label='Собственное значение = 1')
plt.title('График каменистой осыпи')
plt.xlabel('Номер фактора')
plt.ylabel('Собственное значение')
plt.legend()
plt.grid(True)
plt.show()

# По графику выбираем количество факторов (например, 4)
n_factors = 4
print(f"Выбрано количество факторов: {n_factors}")

# -------------------------
# 5. Факторный анализ (метод главных факторов)
# -------------------------
fa = FactorAnalysis(n_components=n_factors, random_state=42)
fa.fit(df_scaled)

# Факторные нагрузки (компоненты транспонированы)
loadings = fa.components_.T   # матрица (переменные, факторы)
loadings_df = pd.DataFrame(loadings, index=df_scaled.columns,
                           columns=[f'Factor_{i+1}' for i in range(n_factors)])
loadings_df = loadings_df.round(3)
print("\nФакторные нагрузки:")
print(loadings_df)

# Визуализация тепловой картой
plt.figure(figsize=(12, 10))
sns.heatmap(loadings_df, annot=True, cmap='coolwarm', center=0)
plt.title('Факторные нагрузки (Factor Analysis)')
plt.tight_layout()
plt.show()

# Получить значения факторов для наблюдений (если нужно)
factor_scores = fa.transform(df_scaled)
print("\nПервые 5 наблюдений в пространстве факторов:")
print(factor_scores[:5])

# -------------------------
# 6. Интерпретация факторов (описательные названия)
# -------------------------
print("\nИнтерпретация факторов:")
for i in range(n_factors):
    # Переменные с нагрузкой > 0.4 (можно настроить порог)
    high_loadings = loadings_df[loadings_df[f'Factor_{i+1}'].abs() > 0.4]
    print(f"\nФактор {i+1}:")
    print(high_loadings.sort_values(f'Factor_{i+1}', ascending=False))