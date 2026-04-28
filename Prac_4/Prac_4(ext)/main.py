import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency
import matplotlib.pyplot as plt

# ----------------------------
# 1. Загрузка данных
# ----------------------------
users = pd.read_csv('users_data.csv')
sessions = pd.read_csv('sessions_data.csv')

print("users_data shape:", users.shape)
print("sessions_data shape:", sessions.shape)
print("\nСтолбцы users_data:", users.columns.tolist())
print("Столбцы sessions_data:", sessions.columns.tolist())

# ----------------------------
# 2. Предобработка
# ----------------------------
# Преобразуем временные метки
sessions['session_start_timestamp'] = pd.to_datetime(sessions['session_start_timestamp'])
sessions['booking_timestamp'] = pd.to_datetime(sessions['booking_timestamp'], errors='coerce')

# Признак наличия бронирования
sessions['has_booking'] = sessions['booking_timestamp'].notna()

# Объединяем с users, чтобы получить группу для каждой сессии
sessions_with_group = sessions.merge(users, on='user_id', how='left')

# ----------------------------
# Гипотеза 1: количество бронирований на пользователя по группам
# ----------------------------
bookings_per_user = sessions_with_group.groupby(['user_id', 'experiment_group'])['has_booking'].sum().reset_index()
bookings_per_user.rename(columns={'has_booking': 'bookings'}, inplace=True)

group_control = bookings_per_user[bookings_per_user['experiment_group'] == 'control']['bookings']
group_variant = bookings_per_user[bookings_per_user['experiment_group'] == 'variant']['bookings']

print(f"\n--- Гипотеза 1: количество бронирований ---")
print(f"Размер выборок: control={len(group_control)}, variant={len(group_variant)}")
print(f"Среднее кол-во бронирований: control={group_control.mean():.2f}, variant={group_variant.mean():.2f}")

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.hist(group_control, bins=30, alpha=0.5, label='control')
plt.hist(group_variant, bins=30, alpha=0.5, label='variant')
plt.legend()
plt.title('Распределение количества бронирований')
plt.subplot(1,2,2)
plt.boxplot([group_control, group_variant], tick_labels=['control','variant'])
plt.title('Ящик с усами')
plt.tight_layout()
plt.show()

t_stat, p_t = ttest_ind(group_control, group_variant, equal_var=False)
u_stat, p_u = mannwhitneyu(group_control, group_variant, alternative='two-sided')

print(f"T-test: statistic={t_stat:.4f}, p-value={p_t:.6f}")
print(f"Mann-Whitney U test: statistic={u_stat:.4f}, p-value={p_u:.6f}")

alpha = 0.05
if p_t < alpha:
    print("T-test: Отвергаем H0 – различия статистически значимы.")
else:
    print("T-test: Не отвергаем H0 – различия не значимы.")
if p_u < alpha:
    print("Mann-Whitney: Отвергаем H0 – различия значимы.")
else:
    print("Mann-Whitney: Не отвергаем H0 – различия не значимы.")

# ----------------------------
# Гипотеза 2: время до бронирования у групп
# ----------------------------
sessions_booked = sessions_with_group[sessions_with_group['time_to_booking'].notna()].copy()
time_control = sessions_booked[sessions_booked['experiment_group'] == 'control']['time_to_booking']
time_variant = sessions_booked[sessions_booked['experiment_group'] == 'variant']['time_to_booking']

print(f"\n--- Гипотеза 2: время до бронирования ---")
print(f"Размер выборок: control={len(time_control)}, variant={len(time_variant)}")
print(f"Среднее время (сек): control={time_control.mean():.2f}, variant={time_variant.mean():.2f}")

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.hist(time_control, bins=30, alpha=0.5, label='control')
plt.hist(time_variant, bins=30, alpha=0.5, label='variant')
plt.legend()
plt.title('Распределение времени до бронирования (сек)')
plt.subplot(1,2,2)
plt.boxplot([time_control, time_variant], tick_labels=['control','variant'])
plt.title('Ящик с усами')
plt.tight_layout()
plt.show()

t_stat2, p_t2 = ttest_ind(time_control, time_variant, equal_var=False)
u_stat2, p_u2 = mannwhitneyu(time_control, time_variant, alternative='two-sided')

print(f"T-test: statistic={t_stat2:.4f}, p-value={p_t2:.6f}")
print(f"Mann-Whitney U test: statistic={u_stat2:.4f}, p-value={p_u2:.6f}")

if p_t2 < alpha:
    print("T-test: Отвергаем H0 – различия статистически значимы.")
else:
    print("T-test: Не отвергаем H0 – различия не значимы.")
if p_u2 < alpha:
    print("Mann-Whitney: Отвергаем H0 – различия значимы.")
else:
    print("Mann-Whitney: Не отвергаем H0 – различия не значимы.")

# ----------------------------
# Гипотеза 3: день/ночь vs бронирование
# ----------------------------
sessions_with_group['start_hour'] = sessions_with_group['session_start_timestamp'].dt.hour
sessions_with_group['day_night'] = sessions_with_group['start_hour'].apply(lambda x: 'day' if 6 <= x <= 21 else 'night')

contingency = pd.crosstab(sessions_with_group['day_night'], sessions_with_group['has_booking'])
print("\n--- Гипотеза 3: день/ночь vs бронирование ---")
print("Таблица сопряженности:")
print(contingency)

chi2, p_dn, dof, expected = chi2_contingency(contingency)
print(f"Хи-квадрат: {chi2:.4f}")
print(f"p-value: {p_dn:.6f}")

if p_dn < alpha:
    print("Вывод: Отвергаем H0 – количество бронирований зависит от времени суток.")
else:
    print("Вывод: Не отвергаем H0 – нет статистически значимой связи между временем суток и бронированиями.")

print("\nВсе тесты завершены.")