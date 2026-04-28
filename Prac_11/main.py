import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

#Зазадние 1 (4 балла)
print("ЗАДАНИЕ 1: Классификация на встроенном датасете breast_cancer")

#1. Выбор данных
print("\n1. Загрузка данных (breast_cancer из sklearn.datasets)...")
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
df_cancer = pd.DataFrame(data.data, columns=data.feature_names)
df_cancer['target'] = data.target
print(f"Размер данных: {df_cancer.shape}")
print(f"Пример записей:\n{df_cancer.head(2)}")
print(f"Распределение классов:\n{df_cancer['target'].value_counts()}")

#2. Подготовка данных (стандартизация, разделение)
print("\n2. Подготовка данных: стандартизация и разделение на train/test...")
X = df_cancer.drop('target', axis=1)
y = df_cancer['target']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#3. Построение модели классификации (RandomForest)
print("\n3. Обучение модели RandomForestClassifier...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

#4. Оценка модели
print("\n4. Оценка точности модели...")
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Матрица ошибок и отчёт
cm = confusion_matrix(y_test, y_pred)
print(f"Точность (accuracy): {accuracy:.4f}")
print("\nОтчёт по классификации:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# Важность признаков
feature_importance = pd.Series(rf_model.feature_importances_, index=data.feature_names).sort_values(ascending=False)

# --- ДИНАМИЧЕСКИЕ ВЫВОДЫ ---
print("\n5. Выводы по заданию №1:")

# 5.1 Оценка качества модели
if accuracy >= 0.95:
    quality_text = "отличное (выше 95%)"
elif accuracy >= 0.85:
    quality_text = "хорошее"
else:
    quality_text = "удовлетворительное, требуется дополнительная настройка"

print(f" - Модель классификации на данных '{data.DESCR.splitlines()[0]}' достигла точности {accuracy:.4f} — это {quality_text} результат.")

# 5.2 Топ-признаки (динамически)
top_n = 3
top_features = feature_importance.head(top_n).index.tolist()
top_values = feature_importance.head(top_n).values
features_str = ", ".join([f"'{f}' ({v:.3f})" for f, v in zip(top_features, top_values)])
print(f" - {top_n} наиболее важных признака(ов): {features_str}.")

# 5.3 Различие классов (используем имена классов из data.target_names)
class_names = data.target_names
print(f" - Модель хорошо различает классы '{class_names[0]}' и '{class_names[1]}' (см. матрицу ошибок).")

# 5.4 Требования к предобработке (зависит от данных, но универсально)
print(" - Для подобных числовых признаков стандартизация полезна, выбросы не критичны (RandomForest устойчив).")
print(" - При необходимости можно дополнительно применить отбор признаков, но текущая точность уже приемлема.\n")

# Визуализации (матрица ошибок, важность признаков) остаются как были
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Матрица ошибок для RandomForest')
plt.ylabel('Истинный класс')
plt.xlabel('Предсказанный класс')
plt.show()

plt.figure()
feature_importance.head(10).plot(kind='bar')
plt.title('Топ-10 важных признаков по модели RandomForest')
plt.ylabel('Важность')
plt.tight_layout()
plt.show()