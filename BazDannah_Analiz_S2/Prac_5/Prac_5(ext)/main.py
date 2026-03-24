import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import seaborn as sns

df = sns.load_dataset('diamonds')

# Создаём фигуру с двумя подграфиками: гистограмма цены и scatter цена vs карат
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Распределение цены', 'Зависимость цены от веса'),
    specs=[[{'type': 'histogram'}, {'type': 'scatter'}]]
)

# Добавляем гистограмму
fig.add_trace(
    go.Histogram(x=df['price'], nbinsx=50, name='Цена', marker_color='steelblue'),
    row=1, col=1
)

# Добавляем scatter plot
fig.add_trace(
    go.Scatter(x=df['carat'], y=df['price'], mode='markers', marker=dict(size=3, opacity=0.5, color='orange'), name='Карат/Цена'),
    row=1, col=2
)

# Обновляем layout
fig.update_layout(
    title_text='Дашборд анализа алмазов',
    width=1200,
    height=500,
    showlegend=False,
    template='plotly_white'
)

fig.show()

