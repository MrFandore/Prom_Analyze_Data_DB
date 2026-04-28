import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import mean_squared_error
from math import sqrt
from pmdarima.arima import auto_arima

plt.rcParams['figure.figsize'] = (12, 6)
sns.set_style('whitegrid')

print("ЗАДАНИЕ 1: Анализ временного ряда (AirPassengers)")
print("\n1. Загрузка датасета...")
try:
    df = pd.read_csv('AirPassengers.csv')
    print("Файл 'AirPassengers.csv' загружен.")
except FileNotFoundError:
    from statsmodels.datasets import get_rdataset
    df = get_rdataset('AirPassengers', 'datasets').data
    print("Загружен встроенный датасет 'AirPassengers'.")
print(f"Размер данных: {df.shape}")
print(df.head())

if 'Month' in df.columns:
    date_col = 'Month'
else:
    date_col = df.columns[0]
if '#Passengers' in df.columns:
    value_col = '#Passengers'
else:
    value_col = df.columns[1]

df[date_col] = pd.to_datetime(df[date_col])
df.set_index(date_col, inplace=True)
series = df[value_col].sort_index()
print(f"Индекс преобразован в datetime. Период: {series.index.min()} – {series.index.max()}")

print("\n2. Визуализация временного ряда...")
plt.figure()
plt.plot(series, color='steelblue')
plt.title('Исходный временной ряд (ежемесячные данные)')
plt.xlabel('Дата')
plt.ylabel(value_col)
plt.grid(True)
plt.show()

print("\n3. Проверка стационарности...")
adf_result = adfuller(series, autolag='AIC')
print(f'ADF statistic: {adf_result[0]:.4f}')
print(f'p-value: {adf_result[1]:.4f}')
if adf_result[1] > 0.05:
    print('Ряд НЕ стационарен (p-value > 0.05) – требуется дифференцирование.')
    stationary_note = 'нестационарен'
else:
    print('Ряд стационарен (p-value <= 0.05).')
    stationary_note = 'стационарен'

print("\n4. Разложение ряда (аддитивная модель)...")
if series.index.inferred_freq is None:
    series = series.asfreq('MS')
decomposition = seasonal_decompose(series, model='additive', period=12)
fig = decomposition.plot()
fig.set_size_inches(12, 9)
plt.tight_layout()
plt.show()

print("\n5. График автокорреляции...")
plot_acf(series, lags=40, alpha=None)
plt.title('Автокорреляционная функция (ACF) исходного ряда')
plt.show()
train = series.iloc[:-12]
test = series.iloc[-12:]
print(f"\nОбучающая выборка: {train.index.min()} – {train.index.max()} (n={len(train)})")
print(f"Тестовая выборка: {test.index.min()} – {test.index.max()} (n={len(test)})")

plt.figure()
plt.plot(train, label='Обучающая', color='steelblue')
plt.plot(test, label='Тестовая', color='red')
plt.title('Разделение на обучающую и тестовую выборки')
plt.legend()
plt.show()

print("\n7. Подбор параметров модели ARIMA (auto_arima)...")
model_arima = auto_arima(
    train,
    start_p=0, max_p=5,
    start_d=0, max_d=2,
    start_q=0, max_q=5,
    seasonal=True, m=12,
    start_P=0, max_P=2,
    start_D=0, max_D=1,
    start_Q=0, max_Q=2,
    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True,
    n_fits=30
)

print("\nЛучшая модель:")
print(model_arima.summary())
model_arima.fit(train)
forecast = model_arima.predict(n_periods=len(test))
forecast = pd.Series(forecast, index=test.index, name='Прогноз')

print("\n8. Оценка качества прогноза...")
mse = mean_squared_error(test, forecast)
rmse = sqrt(mse)
mape = np.mean(np.abs((test - forecast) / test)) * 100
print(f'MSE  = {mse:.2f}')
print(f'RMSE = {rmse:.2f}')
print(f'MAPE = {mape:.2f}%')

plt.figure()
plt.plot(train, label='Обучающая выборка', color='steelblue')
plt.plot(test, label='Фактические значения', color='red')
plt.plot(forecast, label='Прогноз (ARIMA)', color='green', linestyle='--')
plt.title('Сравнение прогноза с фактическими данными')
plt.legend()
plt.grid(True)
plt.show()

print("\n9. ВЫВОДЫ ПО ЗАДАНИЮ 1:")
print(f" - Использован датасет '{value_col}' за период {series.index.year.min()}–{series.index.year.max()}.")
print(f" - Тест Дики-Фуллера показал p-value = {adf_result[1]:.4f} → ряд {stationary_note}.")
print(" - Разложение выявило чёткий восходящий тренд и выраженную сезонность с периодом 12 месяцев.")
print(f" - Лучшая модель по auto_arima: {model_arima.order} с сезонным порядком {model_arima.seasonal_order}.")
print(f" - Качество прогноза на тестовом периоде (12 месяцев): RMSE = {rmse:.2f}, MAPE = {mape:.2f}%.")
if mape < 10:
    print(" - Модель показала высокую точность (MAPE < 10%).")
elif mape < 20:
    print(" - Модель показала хорошую точность (MAPE < 20%).")
else:
    print(" - Точность модели средняя; можно улучшить подбором параметров или использованием SARIMA.")
