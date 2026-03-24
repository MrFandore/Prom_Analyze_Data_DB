import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

# Загрузка данных (укажите правильный путь)
df = pd.read_csv('train_data.csv')

# Предположим, что целевая переменная называется 'age'
target = 'age'
if target not in df.columns:
    raise ValueError(f"Целевая колонка '{target}' не найдена. Проверьте названия: {df.columns.tolist()}")

# Удаляем строки с пропусками в целевой переменной
df = df.dropna(subset=[target])

# Определяем числовые и категориальные признаки
X = df.drop(target, axis=1)
y = df[target]

# Разделяем типы колонок
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

# Создаём предобработчик: числовые оставляем как есть, категориальные кодируем OneHot
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Разделение на обучающую и тестовую
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создаём пайплайн: сначала предобработка, затем линейная регрессия
pipeline_lr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

pipeline_lr.fit(X_train, y_train)
y_pred = pipeline_lr.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f'Linear Regression R² = {r2:.4f}')

# Если R² < 0.8, пробуем полиномиальную регрессию (степень 2)
if r2 < 0.8:
    # Полиномиальную регрессию сложно напрямую включить в пайплайн с OneHot,
    # поэтому создадим отдельный пайплайн с PolynomialFeatures.
    # Однако PolynomialFeatures применяется к числовым признакам, а категориальные нужно оставить как есть.
    # Лучше: сначала применить OneHot, затем PolynomialFeatures к числовым.
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import FeatureUnion

    # Разделим предобработку на числовую и категориальную части
    numeric_transformer = Pipeline(steps=[
        ('poly', PolynomialFeatures(degree=2, include_bias=False))
    ])
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor_poly = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    pipeline_poly = Pipeline(steps=[
        ('preprocessor', preprocessor_poly),
        ('regressor', LinearRegression())
    ])

    pipeline_poly.fit(X_train, y_train)
    y_pred_poly = pipeline_poly.predict(X_test)
    r2_poly = r2_score(y_test, y_pred_poly)
    print(f'Polynomial (degree 2) R² = {r2_poly:.4f}')
    if r2_poly >= 0.8:
        best_pipeline = pipeline_poly
        best_r2 = r2_poly
        best_is_poly = True
    else:
        best_pipeline = pipeline_lr
        best_r2 = r2
        best_is_poly = False
else:
    best_pipeline = pipeline_lr
    best_r2 = r2
    best_is_poly = False

print(f'Best model R² = {best_r2:.4f}')

# Генерация 10 новых наблюдений (случайные значения в пределах диапазонов признаков)
n_new = 10
new_data = {}
for col in X.columns:
    if col in numeric_cols:
        new_data[col] = np.random.uniform(X[col].min(), X[col].max(), n_new)
    else:
        # для категориальных выбираем случайное значение из уникальных
        unique_vals = X[col].dropna().unique()
        new_data[col] = np.random.choice(unique_vals, n_new)
X_new = pd.DataFrame(new_data)

# Предсказание
predictions = best_pipeline.predict(X_new)

print("\nСгенерированные данные и предсказанный возраст:")
result = X_new.copy()
result['predicted_age'] = predictions
print(result)