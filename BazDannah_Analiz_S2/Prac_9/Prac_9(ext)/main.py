# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE   # требуется pip install imbalanced-learn

# -------------------------
# 1. Загрузка данных
# -------------------------
df = pd.read_csv('final_test.csv')   # файл из примера

# -------------------------
# 2. Проверка и очистка пропусков
# -------------------------
print("Исходное количество строк:", len(df))
print("Пропуски в size:", df['size'].isna().sum())

# Удаляем строки с пропущенным размером (целевая переменная)
df = df.dropna(subset=['size'])

# Заполняем пропуски в возрасте и росте медианой
df['age'] = df['age'].fillna(df['age'].median())
df['height'] = df['height'].fillna(df['height'].median())

# -------------------------
# 3. Инженерия признаков
# -------------------------
# Индекс массы тела (рост в метрах)
df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
# Квадрат веса
df['weight_sq'] = df['weight'] ** 2
# Взаимодействие возраста и веса
df['age_weight'] = df['age'] * df['weight']
# Логарифм веса
df['log_weight'] = np.log1p(df['weight'])

# -------------------------
# 4. Кодирование целевой переменной (размеры одежды)
# -------------------------
size_order = {'S': 0, 'M': 1, 'L': 2, 'XL': 3}
df['size_encoded'] = df['size'].map(size_order)

# Удаляем строки, где размер не из списка (если такие есть)
df = df.dropna(subset=['size_encoded'])

# Удаляем любые другие пропуски (если остались)
df = df.dropna()

print("После очистки:", len(df))

# -------------------------
# 5. Подготовка данных для моделирования
# -------------------------
X = df.drop(['size', 'size_encoded'], axis=1)   # все признаки, кроме целевой
y = df['size_encoded']

# Разделение на train/test (стратифицированное по y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Масштабирование числовых признаков
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------
# 6. Балансировка классов с помощью SMOTE
# -------------------------
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

print("Распределение классов после SMOTE:")
print(pd.Series(y_train_bal).value_counts())

# -------------------------
# 7. Поиск лучших гиперпараметров для XGBoost
# -------------------------
xgb = XGBClassifier(random_state=42, eval_metric='mlogloss')

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}

grid = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train_bal, y_train_bal)

print("Лучшие параметры XGBoost:", grid.best_params_)
best_xgb = grid.best_estimator_

# -------------------------
# 8. Обучение других моделей для сравнения
# -------------------------
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_bal, y_train_bal)

gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
gb.fit(X_train_bal, y_train_bal)

# -------------------------
# 9. Оценка точности на тестовой выборке
# -------------------------
models = {
    'XGBoost (optimized)': best_xgb,
    'Random Forest': rf,
    'Gradient Boosting': gb
}

for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name}:")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=size_order.keys(), zero_division=0))

# -------------------------
# 10. Стекинг для улучшения
# -------------------------
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

estimators = [
    ('xgb', best_xgb),
    ('rf', rf),
    ('gb', gb)
]

stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=3
)
stack.fit(X_train_bal, y_train_bal)
y_pred_stack = stack.predict(X_test_scaled)
acc_stack = accuracy_score(y_test, y_pred_stack)
print(f"\nStacking Classifier Accuracy: {acc_stack:.4f}")
print(classification_report(y_test, y_pred_stack, target_names=size_order.keys(), zero_division=0))

# -------------------------
# 11. Вывод лучшего результата
# -------------------------
best_acc = max(acc_stack, accuracy_score(y_test, best_xgb.predict(X_test_scaled)))
print(f"\nДостигнутая точность: {best_acc:.4f}")
if best_acc >= 0.7:
    print("Цель достигнута: точность ≥ 0.7")
else:
    print("Цель не достигнута. Возможные улучшения:")
    print("- Добавить больше признаков (например, произведение возраста и роста)")
    print("- Использовать LightGBM или CatBoost")
    print("- Применить нейросетевой подход (MLPClassifier)")