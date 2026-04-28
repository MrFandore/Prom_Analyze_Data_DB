import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

# Загрузка данных
df = pd.read_csv('penguins_binary_classification.csv')

# Если species содержит более двух классов, оставляем два самых частых
if df['species'].nunique() > 2:
    top_two = df['species'].value_counts().index[:2].tolist()
    df = df[df['species'].isin(top_two)].copy()
    print(f"Оставлены классы: {top_two}")

# Создаём бинарную целевую переменную
df['target'] = (df['species'] == df['species'].unique()[1]).astype(int)

# Определяем признаки (числовые)
# Известные числовые колонки в датасете Palmer Penguins
possible_numeric = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
# Оставляем только те, которые есть в датасете
numeric_cols = [col for col in possible_numeric if col in df.columns]

# Если есть категориальные признаки (например, 'sex', 'island'), их тоже можно закодировать
categorical_cols = [col for col in df.columns if col not in numeric_cols + ['species', 'target'] and df[col].dtype == 'object']

# Удаляем строки с пропусками в признаках и целевой переменной
df = df.dropna(subset=numeric_cols + categorical_cols + ['target'])
X = df[numeric_cols + categorical_cols]
y = df['target']

# Предобработка: числовые масштабируем, категориальные кодируем
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Пайплайн с логистической регрессией
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Разделение на обучающую и тестовую
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Оценка
acc = accuracy_score(y_test, y_pred)
print(f'Accuracy: {acc:.4f}')
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# Визуализация разделения по двум первым числовым признакам
if len(numeric_cols) >= 2:
    X_vis = df[numeric_cols[:2]]
    y_vis = y
    # Обучаем отдельную модель без категориальных признаков для визуализации
    logreg_vis = LogisticRegression()
    logreg_vis.fit(X_vis, y_vis)

    x_min, x_max = X_vis.iloc[:, 0].min() - 1, X_vis.iloc[:, 0].max() + 1
    y_min, y_max = X_vis.iloc[:, 1].min() - 1, X_vis.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    # Создаём DataFrame с именами колонок
    grid_df = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=X_vis.columns)
    Z = logreg_vis.predict(grid_df)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8,6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X_vis.iloc[:, 0], X_vis.iloc[:, 1], c=y_vis, edgecolors='k', cmap='coolwarm')
    plt.xlabel(numeric_cols[0])
    plt.ylabel(numeric_cols[1])
    plt.title('Logistic Regression Decision Boundary')
    plt.show()