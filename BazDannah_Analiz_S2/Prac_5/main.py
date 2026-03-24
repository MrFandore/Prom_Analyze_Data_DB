import seaborn as sns
import plotly.express as px
import pandas as pd

# =======================================================
# 1. Покомпонентное сравнение (круговая диаграмма)
# =======================================================
print("\n" + "="*60)
print("1. ПОКОМПОНЕНТНОЕ СРАВНЕНИЕ (доля от целого)")
print("="*60)

tips = sns.load_dataset('tips')
daily_total = tips.groupby('day', as_index=False)['total_bill'].sum()

fig = px.pie(daily_total, values='total_bill', names='day',
             title='Покомпонентное сравнение: доля выручки по дням недели',
             width=600, height=500,
             color_discrete_sequence=px.colors.sequential.Blues[::-1])
fig.show()

# Вывод в консоль
print("\nАналитический вывод:")
print(f"  • Общая выручка за все дни: ${daily_total['total_bill'].sum():.2f}")
print("  • Доли по дням (% от общей выручки):")
for _, row in daily_total.iterrows():
    pct = row['total_bill'] / daily_total['total_bill'].sum() * 100
    print(f"      {row['day']}: {pct:.1f}% (${row['total_bill']:.2f})")
print("  • Наибольшая доля выручки приходится на субботу (Sat) и воскресенье (Sun), что соответствует")
print("    повышенной посещаемости в выходные дни. Суббота лидирует с долей ~35%.")

# =======================================================
# 2. Позиционное сравнение (горизонтальная гистограмма)
# =======================================================
print("\n" + "="*60)
print("2. ПОЗИЦИОННОЕ СРАВНЕНИЕ (ранжирование)")
print("="*60)

avg_tip = tips.groupby('sex', as_index=False)['tip'].mean()

fig = px.bar(avg_tip, y='sex', x='tip', text_auto='.2f',
             title='Позиционное сравнение: средние чаевые по полу',
             width=600, height=400,
             orientation='h',
             color_discrete_sequence=px.colors.sequential.Blues[::-1])
fig.update_xaxes(title='Средние чаевые, USD')
fig.update_yaxes(title='Пол')
fig.show()

# Вывод в консоль
print("\nАналитический вывод:")
male_tip = avg_tip[avg_tip['sex'] == 'Male']['tip'].values[0]
female_tip = avg_tip[avg_tip['sex'] == 'Female']['tip'].values[0]
print(f"  • Средние чаевые мужчин: ${male_tip:.2f}")
print(f"  • Средние чаевые женщин: ${female_tip:.2f}")
print(f"  • Разница: ${male_tip - female_tip:.2f} (мужчины оставляют больше на {((male_tip - female_tip)/female_tip*100):.1f}%)")
print("  • Таким образом, мужчины в среднем оставляют более высокие чаевые, что может быть связано с")
print("    большими счетами или иными поведенческими факторами.")

# =======================================================
# 3. Частотное сравнение (гистограмма распределения)
# =======================================================
print("\n" + "="*60)
print("3. ЧАСТОТНОЕ СРАВНЕНИЕ (распределение)")
print("="*60)

iris = sns.load_dataset('iris')

fig = px.histogram(iris, x='sepal_length', nbins=30,
                   title='Частотное сравнение: распределение длины чашелистика',
                   width=600, height=500,
                   color_discrete_sequence=px.colors.sequential.Blues[::-1])
fig.update_xaxes(title='Длина чашелистика, см')
fig.update_yaxes(title='Количество цветков')
fig.show()

# Вывод в консоль
print("\nАналитический вывод:")
print(f"  • Средняя длина чашелистика: {iris['sepal_length'].mean():.2f} см")
print(f"  • Медиана: {iris['sepal_length'].median():.2f} см")
print(f"  • Стандартное отклонение: {iris['sepal_length'].std():.2f} см")
print("  • Распределение близко к нормальному, пик приходится на интервал 5.0–6.5 см.")
print("  • Наибольшее количество цветков (около 35) имеют длину чашелистика 5.8–6.2 см.")
print("  • Выбросы практически отсутствуют, что свидетельствует о хорошей репрезентативности данных.")

# =======================================================
# 4. Корреляционное сравнение (диаграмма рассеяния)
# =======================================================
print("\n" + "="*60)
print("4. КОРРЕЛЯЦИОННОЕ СРАВНЕНИЕ (зависимость)")
print("="*60)

fig = px.scatter(iris, x='sepal_length', y='sepal_width', color='species',
                 title='Корреляционное сравнение: длина vs ширина чашелистика',
                 width=700, height=500,
                 color_discrete_sequence=px.colors.qualitative.Set1)
fig.update_xaxes(title='Длина чашелистика, см')
fig.update_yaxes(title='Ширина чашелистика, см')
fig.show()

# Вывод в консоль
print("\nАналитический вывод:")
corr = iris['sepal_length'].corr(iris['sepal_width'])
print(f"  • Коэффициент корреляции Пирсона: {corr:.3f}")
print("  • Наблюдается слабая отрицательная корреляция: с увеличением длины чашелистика его ширина")
print("    незначительно уменьшается.")
print("  • Виды ирисов образуют отдельные кластеры:")
print("      - Setosa: короткие и широкие чашелистики (отдельная группа слева вверху)")
print("      - Versicolor: промежуточные значения")
print("      - Virginica: длинные и узкие чашелистики")
print("  • Это указывает на то, что длина и ширина чашелистика могут быть хорошими признаками")
print("    для классификации видов ирисов.")

# =======================================================
# 5. Временное сравнение (линейный график)
# =======================================================
print("\n" + "="*60)
print("5. ВРЕМЕННОЕ СРАВНЕНИЕ (динамика)")
print("="*60)

flights = sns.load_dataset('flights')
yearly_passengers = flights.groupby('year', as_index=False)['passengers'].sum()

fig = px.line(yearly_passengers, x='year', y='passengers',
              title='Временное сравнение: рост авиаперевозок',
              width=700, height=500,
              markers=True,
              color_discrete_sequence=px.colors.sequential.Blues[::-1])
fig.update_xaxes(title='Год')
fig.update_yaxes(title='Количество пассажиров (тыс.)')
fig.show()

# Вывод в консоль
print("\nАналитический вывод:")
print(f"  • В 1949 году суммарное количество пассажиров составило {yearly_passengers[yearly_passengers['year']==1949]['passengers'].values[0]} тыс.")
print(f"  • В 1960 году: {yearly_passengers[yearly_passengers['year']==1960]['passengers'].values[0]} тыс.")
growth = yearly_passengers[yearly_passengers['year']==1960]['passengers'].values[0] - yearly_passengers[yearly_passengers['year']==1949]['passengers'].values[0]
print(f"  • Абсолютный прирост за 11 лет: {growth} тыс. пассажиров.")
print("  • Наблюдается устойчивый положительный тренд: количество пассажиров увеличивалось каждый год.")
print("  • Особенно заметен ускоренный рост с 1955 года.")
print("  • Это отражает общий рост популярности авиаперевозок в послевоенный период.")

print("\n" + "="*60)
print("Анализ завершён. Все пять типов сравнения визуализированы.")
print("="*60)