import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

df = pd.read_csv('customer_data_with_churn.csv')
print(df.head())
print(df.info())

sns.histplot(df['Annual_Income'], kde=True)
plt.title('Распределение годового дохода')
plt.show()

print("_________________"'\n')
sample_for_test = df['Annual_Income'].sample(min(5000, len(df)), random_state=42)
stat, p_value = stats.shapiro(sample_for_test)
print(f'Статистика Шапиро-Уилка: {stat:.3f}, p-value: {p_value:.3f}')

print("_________________"'\n')
alpha = 0.05
if p_value > alpha:
    print('Нет оснований отвергнуть гипотезу о нормальности (данные распределены нормально)')
else:
    print('Отвергаем гипотезу о нормальности (распределение не является нормальным)')
mean_income_by_gender = df.groupby('Gender')['Annual_Income'].mean()
print(mean_income_by_gender)

print("_________________"'\n')
male_income = df[df['Gender'] == 'Male']['Annual_Income'].dropna()
female_income = df[df['Gender'] == 'Female']['Annual_Income'].dropna()
stat, p_value = stats.mannwhitneyu(male_income, female_income, alternative='two-sided')
print(f'U-statistic: {stat:.3f}, p-value: {p_value:.3f}''\n')
if p_value < 0.05:
    print('Различия статистически значимы (отвергаем гипотезу о равенстве распределений)')
else:
    print('Статистически значимых различий нет')

sns.boxplot(x='Gender', y='Annual_Income', data=df)
plt.title('Распределение годового дохода по полу')
plt.show()
print("_________________"'\n')

#Part_2
levels = df['Membership_Level'].unique()
for level in levels:
    data = df[df['Membership_Level'] == level]['Annual_Income'].dropna()
    if len(data) > 3 and len(data) <= 5000:
        stat, p = stats.shapiro(data)
        print(f'Уровень {level}: p-value = {p:.4f}')
    else:
        print(f'Уровень {level}: размер выборки {len(data)} – тест Шапиро-Уилка не применяется')
print("_________________"'\n')
from scipy.stats import levene
groups = [df[df['Membership_Level'] == level]['Annual_Income'].dropna() for level in levels]
stat, p = levene(*groups)
print(f'Levene test: statistic={stat:.3f}, p-value={p:.3f}')
if p < 0.05:
    print('Дисперсии различаются статистически значимо')
else:
    print('Дисперсии можно считать равными')

print("_________________"'\n')
h_stat, p_kw = stats.kruskal(*groups)
print(f'Kruskal-Wallis: H={h_stat:.3f}, p-value={p_kw:.3f}')
if p_kw < 0.05:
    print('Существуют статистически значимые различия между уровнями членства')
else:
    print('Нет статистически значимых различий')

sns.boxplot(x='Membership_Level', y='Annual_Income', data=df)
plt.title('Зависимость годового дохода от уровня членства')
plt.show()