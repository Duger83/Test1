import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

print("Чтение данных...")

df_train = pd.read_csv('train.csv')

cols = list(df_train.drop('id', axis=1).columns)

f = open('nulls.csv', 'w')
print('Признак,Доля_пропусков', file = f) 

for col in cols:
    stats = [
        df_train[col].isnull().mean(),
    ]
    s = col + ','
    for i in stats:
        s = s + str(i) +','
    s = s[:-1]    
    print(s, file = f)
f.close()
print('Cоздан файл c вычислением доли пропусков по каждому признаку - nulls.csv')
#более 1% пропусков обнаружено в 9-и столбцах, удаляем пропуски

df_cleaned = df_train.dropna().drop('id', axis=1)
df_cleaned['Policy Start Date'] = pd.to_numeric(pd.to_datetime(df_cleaned['Policy Start Date']))
#после удаления пропусков осталось 384k строк, это 32% от исходных данных, но и на них SVR у меня обучался более часа...

#категориальные столбцы удаляем - это вместо методов уменьшения размерности 
#иначе SVR не обучится вовсе
cat_cols = [
    'Gender',
    'Marital Status',
    'Education Level',
    'Occupation',
    'Location',
    'Policy Type',
    'Customer Feedback',
    'Smoking Status',
    'Exercise Frequency',
    'Property Type'
]
df_train = df_cleaned.drop(cat_cols, axis=1).copy()

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
print('Cоздан файл cо статичстическими показателями признаков - stats.csv')
  
#матрица корреляций
correlation_matrix = df_train.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Матрица парных корреляций')
plt.show()
#значимых коррелций нет (доход-кредитный рейтинг не в счет), оставляем все как есть

#нормализуем выбранные данные, задно и выбросы игнорируем 
X_min_max = df_train.copy()
for column in X_min_max.columns: 
	X_min_max[column] = (X_min_max[column] - X_min_max[column].min()) / (X_min_max[column].max() - X_min_max[column].min())	  

X = X_min_max.drop('Premium Amount', axis=1)
y = X_min_max['Premium Amount']

#KNN обучим и оценим на всех оставшихся данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)

knn_model = KNeighborsRegressor()
knn_model.fit(X_train, y_train)
knn_predictions = knn_model.predict(X_test)
knn_mse = mean_squared_error(y_test, knn_predictions)
knn_mae = mean_absolute_error(y_test, knn_predictions)
knn_r2 = r2_score(y_test, knn_predictions)

# SVR обучим и оценим на случайной выборке примерно в 70k строк... для ускорения
X_sample, _, y_sample, _ = train_test_split(X, y, test_size=0.8, random_state=21)
X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.3, random_state=10)

svr_model = SVR()
svr_model.fit(X_train, y_train)
svr_predictions = svr_model.predict(X_test)
svr_mse = mean_squared_error(y_test, svr_predictions)
svr_mae = mean_absolute_error(y_test, svr_predictions)
svr_r2 = r2_score(y_test, svr_predictions)

# сравнение моделей до подбора гиперпараметров
results = {
    'KNN': {'MSE': knn_mse, 'MAE': knn_mae, 'R^2': knn_r2},
    'SVR': {'MSE': svr_mse, 'MAE': svr_mae, 'R^2': svr_r2}
}

for model, metrics in results.items():
    print(f"{model} - MSE: {metrics['MSE']}, MAE: {metrics['MAE']}, R^2: {metrics['R^2']}")

best_model = min(results, key=lambda x: results[x]['MSE'])
print('Лучшая модель по MSE:', best_model)

# подбор гиперпараметров для SVR
X_sample, _, y_sample, _ = train_test_split(X, y, test_size=0.95, random_state=21) #уменьшаем количество строк в выборке до 18k, иначе не хватает аппаратных ресурсов
X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.3, random_state=10)

param_grid = {
    'kernel': ['poly', 'sigmoid', 'rbf'],
    'C': [0.1, 1, 10],
    'epsilon': [0.01, 0.1, 0.2]
}

grid_search = GridSearchCV(SVR(), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_svr_model = grid_search.best_estimator_
svr_predictions = best_svr_model.predict(X_test)

svr_mse = mean_squared_error(y_test, svr_predictions)
svr_mae = mean_absolute_error(y_test, svr_predictions)
svr_r2 = r2_score(y_test, svr_predictions)

print('Лучшие гиперпараметры для SVR:', grid_search.best_params_)
print('MSE:', svr_mse)
print('MAE:', svr_mae)
print('R^2:', svr_r2)
#после подбора гиперпараметров качество модели незначитетльно улучшилось, но отрицательное значение метриеи R^2 
#красноречиво говорит о том, что полученный с помощью модели прогноз будет далек от идеала