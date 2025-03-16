import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
import joblib
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

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

# выбросы
for col in df_train_new.columns:
    Q3 = df_train_new[col].quantile(0.75)
    Q1 = df_train_new[col].quantile(0.25)
    IQR = Q3 - Q1
    upper = Q3 + 1.5 * IQR
    lower = Q1 - 1.5 * IQR
    df_train_new[col] = df_train_new[col].map(lambda x: pd.NA if x<lower else x if x<=upper else pd.NA) 
df_train_new = df_train_new.dropna()  
X = df_train_new.drop('sex1', axis=1)
print(X.columns)

sns.boxenplot(data=df_train_new)
plt.xticks(rotation=30)
plt.show() 

#подбор гиперпараметров
n_clusters_range = range(2, 11)
#metrics = [] - в моей версии параметр affinity помечен как 'deprecated', поэтому - без него
linkages = ['ward', 'complete', 'average', 'single']

best_score = -1
best_params = {}

for n_clusters in n_clusters_range:
    for linkage_type in linkages:
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_type)
        labels = clustering.fit_predict(X)  
        #оценка качества кластеризации
        if len(set(labels)) > 1: 
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_params = {
                    'n_clusters': n_clusters,
                    'linkage': linkage_type
                }
print("Лучшие гиперпараметры для AgglomerativeClustering:")
print(best_params)
print(f"Лучший коэффициент силуэта для AgglomerativeClustering: {best_score}")

best_clustering = AgglomerativeClustering(
    n_clusters=best_params['n_clusters'],
    linkage=best_params['linkage']
)

#дендрограмма
plt.figure(figsize=(12, 8))  # Увеличение размера фигуры
Z = linkage(X, method=best_params['linkage'])
dendrogram(Z, truncate_mode='level', p=4)  # Обрезка дендрограммы
plt.title('Дендрограмма кластеризации')
plt.xlabel('Объекты')
plt.ylabel('Расстояние')
plt.show()
#по дендрограмме видно, что данные в общем разделились на два класса (у пингвинов два пола, им чужда толерантность совремнного общества)
#однако в одной ветке целых 298 объектов - почему?  Схожие данные говорят о явном наличии подкласса, 
#наверное, нужно рассматриваить еще и возраст пингвинов, и, наверное, схожие данные получены у особей одного возраста,
#конечно, это не аксолотли, у пингвинов пол с возрастом не меняется, но клювы то растут и утолщаются...

#качество кластеризации
df_train_new['cluster'] = best_clustering.fit_predict(X)
adjusted_rand = adjusted_rand_score(df_train_new['sex1'].astype('category').cat.codes, df_train_new['cluster'])
print(f'Adjusted Rand Index: {adjusted_rand}')
print()
#отрицательное число... хм... странный результат - это значит, что объекты кластеризовались плохо 

#сохранение модели
joblib.dump(best_clustering, 'agglomerative_clustering_model.joblib')

#стандартизируем данные (для AgglomerativeClustering я их нормировал, посмотрим, что будет, если шкалировать)
X = df_train[['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']].dropna()
X_scaled = StandardScaler().fit_transform(X)

#gодбор гиперпараметров
best_eps = None
best_min_samples = None
best_silhouette_score = -1

eps_values = np.arange(0.1, 1.1, 0.1)  
min_samples_values = range(2, 11)     

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_scaled)
        if len(set(labels)) > 1:  
            silhouette_avg = silhouette_score(X_scaled, labels)
            if silhouette_avg > best_silhouette_score:
                best_silhouette_score = silhouette_avg
                best_eps = eps
                best_min_samples = min_samples

print(f"Лучшие гиперпараметры для DBSCAN: eps = {best_eps}, min_samples = {best_min_samples}")
print(f"Лучший коэффициент силуэта для DBSCAN: {best_silhouette_score}")
#коэффициент силуэта для DBSCAN (0.72) существенно выше, чем для AgglomerativeClustering (0.52),
#значит DBSCAN лучше кластеризует предложенные данные о пингвинах