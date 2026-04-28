import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Подавляем предупреждения matplotlib о шрифтах
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Загрузка данных
df = pd.read_csv('song_data.csv')  # укажите правильный путь

# Предположим, целевая переменная называется 'song_popularity'
target = 'song_popularity'
if target not in df.columns:
    raise ValueError(f"Целевая колонка '{target}' не найдена. Доступные колонки: {df.columns.tolist()}")

df = df.dropna()  # удаляем строки с пропусками

# Разделяем признаки
X = df.drop(target, axis=1)
y = df[target]

# Определяем числовые и категориальные колонки
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

# 1. Проверка мультиколлинеарности (только для числовых признаков)
if len(numeric_cols) > 1:
    plt.figure(figsize=(12,8))
    sns.heatmap(X[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix (numeric features)')
    plt.show()

    def calculate_vif(X_df):
        vif_data = pd.DataFrame()
        vif_data['feature'] = X_df.columns
        vif_data['VIF'] = [variance_inflation_factor(X_df.values, i) for i in range(X_df.shape[1])]
        return vif_data

    vif_df = calculate_vif(X[numeric_cols])
    print("VIF values (numeric features):")
    print(vif_df)

# 2. Подготовка данных для моделей
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Предобработка: числовые масштабируем, категориальные кодируем
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# 3. Модели в пайплайнах
# Линейная регрессия
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])
lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)
r2_lr = r2_score(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)

# Ридж-регрессия
ridge_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Ridge(alpha=1.0))
])
ridge_pipeline.fit(X_train, y_train)
y_pred_ridge = ridge_pipeline.predict(X_test)
r2_ridge = r2_score(y_test, y_pred_ridge)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)

# Случайный лес
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])
rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)

# 4. Сравнение
results = pd.DataFrame({
    'Model': ['Linear Regression', 'Ridge Regression', 'Random Forest'],
    'R²': [r2_lr, r2_ridge, r2_rf],
    'MSE': [mse_lr, mse_ridge, mse_rf],
    'MAE': [mae_lr, mae_ridge, mae_rf]
})
print("\nComparison of models:")
print(results)

best_model_name = results.loc[results['R²'].idxmax(), 'Model']
print(f"\nBest model based on R²: {best_model_name}")

# Вывод важности признаков для лучшей модели (если Random Forest) в виде текста, без графика
if best_model_name == 'Random Forest':
    # Получаем обработанные данные
    X_train_processed = preprocessor.fit_transform(X_train)
    # Извлекаем имена признаков
    numeric_features = numeric_cols
    if categorical_cols:
        cat_encoder = preprocessor.named_transformers_['cat']
        cat_features = cat_encoder.get_feature_names_out(categorical_cols).tolist()
    else:
        cat_features = []
    feature_names = numeric_features + cat_features
    # Обучаем отдельный случайный лес для получения важностей
    rf_imp = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_imp.fit(X_train_processed, y_train)
    importances = pd.Series(rf_imp.feature_importances_, index=feature_names).sort_values(ascending=False)
    print("\nFeature Importances (Random Forest):")
    print(importances.head(10))