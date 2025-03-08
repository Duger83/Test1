import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE

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

sns.boxenplot(data=df_train_new)
plt.xticks(rotation=30)
plt.show() 

features = df_train_new.drop('sex1', axis=1)
labels = df_train_new['sex1']

#исходные данные
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df_train_new['culmen_length_mm'], y=df_train_new['culmen_depth_mm'], hue=df_train_new['sex1'], palette='viridis', alpha=0.7)
plt.title('Исходные данные')
plt.xlabel('Длина клюва (мм)')
plt.ylabel('Глубина клюва (мм)')
plt.show()

#ядра для Kernel PCA
kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']
plt.figure(figsize=(15, 10))

for i, kernel in enumerate(kernels):
    kpca = KernelPCA(kernel=kernel, n_components=2)
    features_kpca = kpca.fit_transform(features)
    
    #дисперсия и lost_variance для линейного ядра
    if kernel == 'linear':
        explained_variance = np.var(features_kpca, axis=0)
        lost_variance = 1 - np.sum(explained_variance) / np.sum(np.var(features, axis=0))
        print(f"Kernel: {kernel}, Variance: {explained_variance}, Lost Variance: {lost_variance}")
    #сумма полученных значений дисперсии по каждой компоненте чуть менее 13% - это значит,
    #что при уменьшении размерности большая часть имеющейся в данных информации не была сохранена;
    #значение lost_variance в 11% вполне приемлемо при уменьшении размерности;
    #видимо, с полченными данными работать можно в задачах классификации, а вот в задачах прогнозирования - скорее, нет.

    #графики для PCA
    plt.subplot(2, 3, i + 1)
    sns.scatterplot(x=features_kpca[:, 0], y=features_kpca[:, 1], hue=labels, palette='viridis', alpha=0.7)
    plt.title(f'PCA with {kernel} kernel')

#t-SNE
tsne = TSNE(n_components=2, random_state=42)
features_tsne = tsne.fit_transform(features)

#график для t-SNE
plt.subplot(2, 3, len(kernels) + 1)
sns.scatterplot(x=features_tsne[:, 0], y=features_tsne[:, 1], hue=labels, palette='viridis', alpha=0.7)
plt.title('t-SNE')

plt.tight_layout()
plt.show()
#в итоге можно сказать следующе: ядра модели PCA 'linear', 'poly' и 'rbf' показали похожие друг на друга результаты, и визуализация
#применения этих ядер в общем похожа на исходные данные, значит в данном случае можно применить PCA с одним из этих ядер на выбор;
#ядра 'sigmoid' и 'cosine', а с ними и модель t-SNE, тоже показали похожие результаты, но заметно отличающиеся от исходных данных, 
#скорее всего это объясняется тем, что у нас изначально была небольшая размерность, а эти ядра, и тем более t-SNE, 
#лучше применять, когда в исходных данных действительно большое число признаков.