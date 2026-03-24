import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import os

# ------------------------------
# Настройки
# ------------------------------
DATA_FILE = "segmentation_data.csv"
PLOTS_DIR = "plots"          # папка для сохранения графиков
os.makedirs(PLOTS_DIR, exist_ok=True)

# ------------------------------
# 1. Загрузка данных
# ------------------------------
print("Загрузка данных...")
df = pd.read_csv(DATA_FILE)
print(f"Размер данных: {df.shape}")
print("\nПервые 5 строк:")
print(df.head())

# ------------------------------
# 2. Предобработка
# ------------------------------
# Удаление пропусков
df = df.dropna()

# Выбираем числовые колонки (исключаем идентификаторы, если есть)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'CustomerID' in numeric_cols:
    numeric_cols.remove('CustomerID')
if 'ID' in numeric_cols:
    numeric_cols.remove('ID')

X = df[numeric_cols]
print(f"\nПризнаки для кластеризации: {numeric_cols}")

# Масштабирование
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------
# 3. Определение оптимального числа кластеров
# ------------------------------
print("\nОпределение оптимального числа кластеров...")
inertia = []
sil_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Графики
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(K_range, inertia, 'bo-')
plt.xlabel('Число кластеров k')
plt.ylabel('Инерция')
plt.title('Метод локтя')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(K_range, sil_scores, 'ro-')
plt.xlabel('Число кластеров k')
plt.ylabel('Средний силуэт')
plt.title('Оценка силуэта')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'elbow_silhouette.png'), dpi=150)
plt.show()

# Выбор оптимального k (например, где изгиб и максимум силуэта)
# Можно взять значение, при котором силаэт максимален
optimal_k = K_range[np.argmax(sil_scores)]
print(f"Рекомендуемое число кластеров по силуэту: {optimal_k}")

# ------------------------------
# 4. Кластеризация методом K-means
# ------------------------------
print("\nВыполнение K-means кластеризации...")
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
df['Cluster'] = clusters

# ------------------------------
# 5. Визуализация кластеров с помощью PCA
# ------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=200, c='red', marker='X', label='Центры кластеров')
plt.colorbar(scatter)
plt.title('Кластеры клиентов (PCA)')
plt.xlabel('Первая главная компонента')
plt.ylabel('Вторая главная компонента')
plt.legend()
plt.savefig(os.path.join(PLOTS_DIR, 'pca_clusters.png'), dpi=150)
plt.show()

# ------------------------------
# 6. Профили кластеров
# ------------------------------
print("\nСредние значения признаков по кластерам:")
cluster_profile = df.groupby('Cluster')[numeric_cols].mean()
print(cluster_profile.round(2))

# Визуализация профиля
plt.figure(figsize=(12, 6))
cluster_profile.T.plot(kind='bar', colormap='viridis')
plt.title('Средние значения признаков в кластерах')
plt.ylabel('Среднее значение')
plt.xticks(rotation=45)
plt.legend(title='Кластер')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'cluster_profiles.png'), dpi=150)
plt.show()

# ------------------------------
# 7. Альтернативные методы
# ------------------------------
print("\nПробуем иерархическую кластеризацию...")
agg = AgglomerativeClustering(n_clusters=optimal_k)
agg_labels = agg.fit_predict(X_scaled)

print("\nПробуем DBSCAN...")
# Подбор параметров eps и min_samples можно улучшить, здесь простой вариант
dbscan = DBSCAN(eps=0.5, min_samples=5)
db_labels = dbscan.fit_predict(X_scaled)
n_db_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
print(f"DBSCAN выделил {n_db_clusters} кластеров, шумовых точек: {sum(db_labels == -1)}")

# Сравнение качества
print("\nСравнение метрик:")
print(f"K-means силуэт: {silhouette_score(X_scaled, clusters):.3f}")
print(f"Agglomerative силуэт: {silhouette_score(X_scaled, agg_labels):.3f}")
if n_db_clusters > 1:
    print(f"DBSCAN силуэт: {silhouette_score(X_scaled, db_labels):.3f}")
else:
    print("DBSCAN не выделил значимых кластеров")

# ------------------------------
# 8. Вывод интерпретации
# ------------------------------
print("\n=== Интерпретация кластеров (на основе средних) ===")
for cluster_id in sorted(cluster_profile.index):
    print(f"\nКластер {cluster_id}:")
    # Выводим признаки, где значения выше/ниже общего среднего
    overall_mean = X[numeric_cols].mean()
    diff = cluster_profile.loc[cluster_id] - overall_mean
    high_features = diff[diff > 0].sort_values(ascending=False).head(3)
    low_features = diff[diff < 0].sort_values().head(3)
    if not high_features.empty:
        print("  Высокие значения:", ", ".join([f"{f} ({high_features[f]:.2f} выше среднего)" for f in high_features.index]))
    if not low_features.empty:
        print("  Низкие значения:", ", ".join([f"{f} ({abs(low_features[f]):.2f} ниже среднего)" for f in low_features.index]))