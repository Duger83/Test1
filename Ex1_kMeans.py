import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import joblib

print("Чтение данных...")

df_train = pd.read_csv('penguins.csv')

def sex(s1):
    if s1 == 'MALE': 
        return 1
    if s1 == 'FEMALE':
        return 0
    if s1 == 'NA': 
        return 'NA'

df_train['sex1'] = df_train['sex'].apply(sex)    
df_train = df_train.drop('sex', axis=1)
cols = list(df_train.columns)

f = open('stats.csv', 'w')
print('Признак,Доля_пропусков,Максимум,Минимум,Среднее,Медиана,Дисперсия,Квантиль_0.1,Квантиль_0.9,Квартиль_1,Квартиль_3', file = f)

for col in cols:
    stats = [
        df_train[col].isnull().mean(),
        df_train[col].max(),
        df_train[col].min(),
        df_train[col].mean(),
        df_train[col].median(),
        df_train[col].var(),
        df_train[col].quantile(0.1),
        df_train[col].quantile(0.9),
        df_train[col].quantile(0.25),
        df_train[col].quantile(0.75)
    ]
    s = col + ','
    for i in stats:
        s = s + str(i) +','
    s = s[:-1]    
    print(s, file = f)

f.close()
print('Cоздан файл со статистикой - stats.csv')

#в данных есть пропуски, удалим их
df_train_new = df_train.dropna()

#нормируем данные
X_min_max = df_train_new.copy()
for column in X_min_max.columns: 
	X_min_max[column] = (X_min_max[column] - X_min_max[column].min()) / (X_min_max[column].max() - X_min_max[column].min())	  
df_train_new = X_min_max.copy()

#выбросы
for col in df_train_new.columns:
    Q3 = df_train_new[col].quantile(0.75)
    Q1 = df_train_new[col].quantile(0.25)
    IQR = Q3 - Q1
    upper = Q3 + 1.5 * IQR
    lower = Q1 - 1.5 * IQR
    df_train_new[col] = df_train_new[col].map(lambda x: pd.NA if x<lower else x if x<=upper else pd.NA) 
df_train_new = df_train_new.dropna()  
X = df_train_new.drop('sex1', axis=1)

sns.boxenplot(data=df_train_new)
plt.xticks(rotation=30)
plt.show() 

# Подбор гиперпараметров KMeans
best_n_clusters = 0
best_score = -1
scores = []


# Подбор гиперпараметров
best_score_s = -1
best_model_s = None
best_score_i = -1
best_model_i = None
scores = []
silhouette_scores = []

for n_clusters in range(2, 10):
    for init in ['k-means++', 'random']:
        for max_iter in [100, 300]:
            for algorithm in ['lloyd', 'elkan']:
                kmeans = KMeans(n_clusters=n_clusters, init=init, max_iter=max_iter, algorithm=algorithm, random_state=42)
                kmeans.fit(X)
                
                # Оценка качества кластеризации
                inertia = kmeans.inertia_
                silhouette_avg = silhouette_score(X, kmeans.labels_)

                scores.append((n_clusters, init, max_iter, algorithm, inertia))
                silhouette_scores.append((n_clusters, init, max_iter, algorithm, silhouette_avg))

                if silhouette_avg > best_score_s:
                    best_score_s = silhouette_avg
                    best_model_s = kmeans
                    
                if inertia > best_score_i:
                    best_score_i = inertia
                    best_model_i = kmeans    

#результаты оценки моделей после подбора гиперпараметров
print(f"Best Inertia Score: {best_score_i}")
print(f"Best model is: {best_model_i}", 'init =', best_model_i.init, ' algorithm =', best_model_i.algorithm)   
print()
print(f"Best Silhouette Score: {best_score_s}")
print(f"Best model is: {best_model_s}", 'init =', best_model_s.init, ' algorithm =', best_model_s.algorithm)
#оба метода указали на одни и те же згначения гиперпараметров
# KMeans(max_iter=100, n_clusters=2, random_state=42, init=k-means++, algorithm=lloyd)
#в дальнейшем будем использовать эти значения для визуализации, кроме количества кластеров;
#полученный коэффициент силуэта = 0.52 говорит о том, что кластеризация проведена на среднем уровне - могло быть хуже, но можно и лучше.

#отрисовка оценки методом локтя
inertia = []
for n_clusters in range(2, 10):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, algorithm='lloyd', random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(2, 10), inertia, marker='o')
plt.title('Метод локтя')
plt.xlabel('Количество кластеров')
plt.ylabel('Инерция')
plt.grid()
plt.show()

#отрисовка оценки методом силуэта
plt.figure(figsize=(10, 6))
for n_clusters in range(2, 10):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, algorithm='lloyd', random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    plt.plot(n_clusters, silhouette_avg, 'bo')
plt.title('Метод силуэта')
plt.xlabel('Количество кластеров')
plt.ylabel('Силуэт')
plt.grid()
plt.show()

#отрисовка кластеров
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=100, algorithm='lloyd', random_state=42)
kmeans.fit(X)
labels = kmeans.labels_

plt.figure(figsize=(10, 6))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k')
centers = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Центры кластеров')
plt.title('Визуализация кластеров')
plt.xlabel('Признак_1')
plt.ylabel('Признак_2')
plt.legend()
plt.grid()
plt.show()

#cохранение модели
joblib.dump(kmeans, 'kmeans_model.joblib')