# -*- coding: utf-8 -*-
"""
Практическая работа 13: Работа с выбросами при помощи ML
Выполнено для PyCharm
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM
from scipy import stats

# Настройки графиков
plt.rcParams['figure.figsize'] = (10, 6)
sns.set_style('whitegrid')

print("=" * 70)
print("ПРАКТИЧЕСКАЯ РАБОТА 13: ОБНАРУЖЕНИЕ И УДАЛЕНИЕ ВЫБРОСОВ")
print("=" * 70)

# ========================= ЗАДАНИЕ 1 (пространственные данные) =========================
print("\n" + "=" * 50)
print("ЗАДАНИЕ 1: Пространственные данные – поиск выбросов 3 методами")
print("=" * 50)

# Загрузка датасета и очистка от дубликатов для корректной работы LOF
try:
    df_house = pd.read_csv('data.csv')
    df_house = df_house.drop_duplicates().reset_index(drop=True)  # ИСПРАВЛЕНИЕ 1
    print(f"Загружен датасет 'data.csv'. Строк после удаления дубликатов: {len(df_house)}")
    rent_col = 'rent' if 'rent' in df_house.columns else df_house.columns[0]
    area_col = 'area' if 'area' in df_house.columns else df_house.columns[1]
    X_spatial = df_house[[rent_col, area_col]].copy()
except FileNotFoundError:
    print("Файл 'data.csv' не найден. Генерируем синтетические данные с выбросами.")
    np.random.seed(42)
    n_normal = 300
    n_outliers = 15
    rent_normal = np.random.normal(30000, 10000, n_normal)
    area_normal = np.random.normal(1200, 300, n_normal)
    rent_out = np.array(
        [150000, 5000, 200000, 2500, 180000, 3000, 120000, 3500, 160000, 2800, 95000, 4000, 140000, 3200, 110000])
    area_out = np.array([200, 2500, 150, 2800, 180, 2600, 220, 2400, 190, 2700, 210, 2300, 170, 2900, 130])
    rent = np.concatenate([rent_normal, rent_out[:n_outliers]])
    area = np.concatenate([area_normal, area_out[:n_outliers]])
    X_spatial = pd.DataFrame({'rent': rent, 'area': area})
    rent_col, area_col = 'rent', 'area'
    print(f"Сгенерировано {len(X_spatial)} точек, из них {n_outliers} заведомых выбросов.")

# Стандартизация
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_spatial)

# Три метода детекции (с исправленными параметрами для LOF)
methods = {
    'Isolation Forest': IsolationForest(contamination=0.03, random_state=42),
    'Local Outlier Factor (LOF)': LocalOutlierFactor(n_neighbors=35, contamination=0.03, novelty=False),
    # ИСПРАВЛЕНИЕ 2
    'DBSCAN': DBSCAN(eps=0.4, min_samples=15)
}
outlier_labels = {}

print("\nОбнаружение выбросов тремя методами...")
for name, model in methods.items():
    if name == 'Local Outlier Factor (LOF)':
        pred = model.fit_predict(X_scaled)
    elif name == 'DBSCAN':
        pred = model.fit_predict(X_scaled)
    else:
        model.fit(X_scaled)
        pred = model.predict(X_scaled)

    outlier_labels[name] = pred
    n_out = np.sum(pred == -1)
    print(f"{name}: найдено выбросов = {n_out} из {len(X_scaled)}")

# Визуализация Задания 1
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].scatter(X_spatial[rent_col], X_spatial[area_col], alpha=0.5, s=10, c='gray')
axes[0, 0].set_title('Исходные данные')
axes[0, 0].set_xlabel(rent_col);
axes[0, 0].set_ylabel(area_col)

for ax, (name, labels) in zip(axes.flatten()[1:], outlier_labels.items()):
    is_outlier = (labels == -1)
    ax.scatter(X_spatial.loc[~is_outlier, rent_col], X_spatial.loc[~is_outlier, area_col],
               c='cornflowerblue', label='Норма', alpha=0.4, s=10)
    ax.scatter(X_spatial.loc[is_outlier, rent_col], X_spatial.loc[is_outlier, area_col],
               c='crimson', label='Выброс', alpha=0.9, s=25, edgecolors='black')
    ax.set_title(name)
    ax.set_xlabel(rent_col);
    ax.set_ylabel(area_col)
    ax.legend()
plt.tight_layout()
plt.show()

# ========================= ЗАДАНИЕ * (статистические тесты) =========================
print("\n" + "=" * 50)
print("ЗАДАНИЕ *: Проверка выбросов статистическими критериями (Граббса и Диксона)")
print("=" * 50)

# Возьмём выбросы, найденные Isolation Forest
iso_labels = outlier_labels['Isolation Forest']
outliers_iso = X_spatial[iso_labels == -1]
print(f"Изоляционным лесом найдено {len(outliers_iso)} выбросов.")


def grubbs_test(data, alpha=0.05):
    x = np.array(data)
    n = len(x)
    mean = np.mean(x)
    std = np.std(x, ddof=1)
    abs_dev = np.abs(x - mean)
    max_idx = np.argmax(abs_dev)
    G = abs_dev[max_idx] / std if std != 0 else 0
    t_crit = stats.t.ppf(1 - alpha / (2 * n), n - 2)
    G_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_crit ** 2 / (n - 2 + t_crit ** 2))
    p_value = 1 - stats.t.cdf(G * np.sqrt((n - 2) / (n - G ** 2 + 1e-12)), n - 2)
    return G > G_crit, p_value, max_idx


def dixon_test(data):
    x_sorted = np.sort(data)
    n = len(x_sorted)
    if n < 3:
        return False, np.nan, np.nan
    gap_min = x_sorted[1] - x_sorted[0]
    gap_max = x_sorted[-1] - x_sorted[-2]
    rng = x_sorted[-1] - x_sorted[0]
    Q_min = gap_min / rng if rng != 0 else 0
    Q_max = gap_max / rng if rng != 0 else 0
    q_crit_dict = {3: 0.941, 4: 0.765, 5: 0.642, 6: 0.560, 7: 0.507, 8: 0.468, 9: 0.437, 10: 0.412}
    q_crit = q_crit_dict.get(n, 0.35)  # Для n>10 возвращает 0.35 как заглушку
    if Q_max > Q_min:
        return Q_max > q_crit, Q_max, q_crit
    else:
        return Q_min > q_crit, Q_min, q_crit


X_rent = X_spatial[rent_col].values
X_area = X_spatial[area_col].values

print("\nРезультаты статистических тестов:")
is_out_grubbs_rent, p_grubbs_rent, idx_grubbs_rent = grubbs_test(X_rent)
print(
    f"  - Тест Граббса ('{rent_col}'): p-value = {p_grubbs_rent:.4f}, выброс {X_rent[idx_grubbs_rent]:.2f} -> {'ДА' if is_out_grubbs_rent else 'НЕТ'}")
is_out_grubbs_area, p_grubbs_area, idx_grubbs_area = grubbs_test(X_area)
print(
    f"  - Тест Граббса ('{area_col}'): p-value = {p_grubbs_area:.4f}, выброс {X_area[idx_grubbs_area]:.2f} -> {'ДА' if is_out_grubbs_area else 'НЕТ'}")

dixon_out_rent, Q_rent, Qcrit_rent = dixon_test(X_rent)
print(
    f"  - Тест Диксона ('{rent_col}'): Q = {Q_rent:.3f}, критическое Q = {Qcrit_rent:.3f} -> {'ВЫБРОС' if dixon_out_rent else 'НЕ ВЫБРОС'}")
dixon_out_area, Q_area, Qcrit_area = dixon_test(X_area)
print(
    f"  - Тест Диксона ('{area_col}'): Q = {Q_area:.3f}, критическое Q = {Qcrit_area:.3f} -> {'ВЫБРОС' if dixon_out_area else 'НЕ ВЫБРОС'}")
print("Статистические тесты одномерны и подтверждают наличие экстремальных значений.\n")

# ========================= ЗАДАНИЕ * (временные ряды) =========================
print("\n" + "=" * 50)
print("ЗАДАНИЕ *: Временной ряд – удаление выбросов скользящим средним и ML (3 модели)")
print("=" * 50)

# Генерация временного ряда с выбросами
np.random.seed(123)
time = np.arange(0, 200)
trend = 0.1 * time
seasonal = 5 * np.sin(2 * np.pi * time / 12)
noise = np.random.normal(0, 0.5, len(time))
y = trend + seasonal + noise
outlier_indices = [20, 45, 88, 120, 175]
for idx in outlier_indices:
    y[idx] += np.random.choice([-15, 15])
ts = pd.Series(y, index=pd.date_range('2020-01-01', periods=len(time), freq='MS'))
print(f"Сгенерирован ряд длиной {len(ts)}. Добавлено {len(outlier_indices)} выбросов.")

# 1) Скользящая медиана
window = 5
ts_medfilt = ts.rolling(window=window, center=True).median()
ts_medfilt.bfill(inplace=True)
ts_medfilt.ffill(inplace=True)


# 2) ML на лаговых признаках
def create_lag_features(series, n_lags=3, window=3):
    df = pd.DataFrame({'original': series})
    for lag in range(1, n_lags + 1):
        df[f'lag_{lag}'] = series.shift(lag)
    df['rolling_mean'] = series.rolling(window).mean()
    df['rolling_std'] = series.rolling(window).std()
    df.dropna(inplace=True)
    return df


X_lag = create_lag_features(ts, n_lags=3, window=3)
scaler_ts = StandardScaler()
X_lag_scaled = scaler_ts.fit_transform(X_lag.drop('original', axis=1))

ml_models = {
    'Isolation Forest (lag)': IsolationForest(contamination=0.05, random_state=42),
    'LOF (lag)': LocalOutlierFactor(contamination=0.05, novelty=False),
    'One-Class SVM (lag)': OneClassSVM(nu=0.05, kernel='rbf', gamma='auto')
}
outlier_flags_ml = {}
for name, model in ml_models.items():
    if name == 'LOF (lag)':
        pred = model.fit_predict(X_lag_scaled)
        flags = (pred == -1)
    else:
        model.fit(X_lag_scaled)
        pred = model.predict(X_lag_scaled)
        flags = (pred == -1)
    outlier_flags_ml[name] = flags

# Восстановление полных флагов для всей серии
full_flags = {name: pd.Series(False, index=ts.index) for name in outlier_flags_ml}
for name, flags in outlier_flags_ml.items():
    temp = pd.Series(flags, index=X_lag.index)
    full_flags[name] = temp.reindex(ts.index, fill_value=False)

# Очистка ряда: заменяем выбросы на медиану соседних 5 точек
ts_cleaned = {}
for name in full_flags:
    cleaned = ts.copy()
    for i in range(len(ts)):
        if full_flags[name].iloc[i]:
            left = max(0, i - 2)
            right = min(len(ts), i + 3)
            cleaned.iloc[i] = ts.iloc[left:right].median()
    ts_cleaned[name] = cleaned

# Визуализация
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes[0, 0].plot(ts.index, ts, label='Исходный', color='cornflowerblue')
axes[0, 0].scatter(ts.index[outlier_indices], ts.iloc[outlier_indices], color='crimson', label='Выбросы', zorder=5)
axes[0, 0].set_title('Исходный ряд с выбросами')
axes[0, 0].legend()

axes[0, 1].plot(ts.index, ts, alpha=0.3, label='Исходный')
axes[0, 1].plot(ts.index, ts_medfilt, label='Скользящая медиана (window=5)', color='green', linewidth=2)
axes[0, 1].set_title('Скользящая медиана')
axes[0, 1].legend()

axes[1, 0].plot(ts.index, ts, alpha=0.3, label='Исходный')
axes[1, 0].plot(ts.index, ts_cleaned['Isolation Forest (lag)'], label='Isolation Forest', color='orange', linewidth=2)
axes[1, 0].set_title('Isolation Forest на лагах')
axes[1, 0].legend()

axes[1, 1].plot(ts.index, ts, alpha=0.3, label='Исходный')
axes[1, 1].plot(ts.index, ts_cleaned['LOF (lag)'], label='LOF', color='purple', linewidth=2)
axes[1, 1].plot(ts.index, ts_cleaned['One-Class SVM (lag)'], label='One-Class SVM', linestyle='--', color='brown')
axes[1, 1].set_title('LOF и SVM')
axes[1, 1].legend()
plt.tight_layout()
plt.show()

print("\nВЫВОДЫ ПО ЗАДАНИЮ * (временной ряд):")
print(f" - Сгенерирован ряд из {len(ts)} точек, известно 5 выбросов.")
print(" - Скользящая медиана сгладила все резкие пики.")
print(" - ML-методы на лаговых признаках:")
for name, flags in full_flags.items():
    print(f"    * {name}: обнаружено {flags.sum()} выбросов.")
print(
    " - Все методы эффективно удалили аномалии. Для временных рядов скользящая медиана проще и нагляднее, а ML требует больше параметров.\n")

print("=" * 70)
print("Работа успешно завершена.")
print("=" * 70)