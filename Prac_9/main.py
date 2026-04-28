# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# -------------------------
# 1. Загрузка и первичный осмотр данных
# -------------------------
df = pd.read_csv('weather_classification_data.csv')
print("Размер данных:", df.shape)
print("\nПервые 5 строк:")
print(df.head())
print("\nИнформация о данных:")
print(df.info())
print("\nСтатистика:")
print(df.describe())

# Целевая переменная: 'Weather Type'
# Проверим распределение классов
print("\nРаспределение классов:")
print(df['Weather Type'].value_counts())
sns.countplot(x='Weather Type', data=df)
plt.title('Распределение типов погоды')
plt.xticks(rotation=45)
plt.show()

# -------------------------
# 2. Предобработка данных
# -------------------------
# Преобразуем категориальные признаки в числовые
categorical_cols = ['Cloud Cover', 'Season', 'Location', 'Weather Type']
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Отделяем признаки и целевую переменную
X = df.drop('Weather Type', axis=1)
y = df['Weather Type']

# Разделяем на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Масштабирование признаков (для SVM и kNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------
# 3. Обучение и оценка моделей
# -------------------------
models = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(kernel='rbf', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss')
}

results = {}

for name, model in models.items():
    print(f"\n=== Обучение {name} ===")
    if name in ['KNN', 'SVM']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    results[name] = {
        'accuracy': acc,
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1': report['weighted avg']['f1-score'],
        'report': classification_report(y_test, y_pred, zero_division=0),
        'confusion': confusion_matrix(y_test, y_pred)
    }
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))

# -------------------------
# 4. Сравнение результатов
# -------------------------
comparison = pd.DataFrame({
    name: {metric: results[name][metric] for metric in ['accuracy', 'precision', 'recall', 'f1']}
    for name in models.keys()
}).T
print("\n=== Сравнение метрик ===")
print(comparison)

# Визуализация
comparison.plot(kind='bar', figsize=(10,6))
plt.title('Сравнение моделей классификации')
plt.ylabel('Значение метрики')
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# Вывод лучшей модели
best_model = comparison['f1'].idxmax()
print(f"\nЛучшая модель по F1-score: {best_model}")
print("Отчёт для лучшей модели:")
print(results[best_model]['report'])