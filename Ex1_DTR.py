import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
import numpy as np

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
#после удаления пропусков осталось 384k строк, это 32% от исходных данных

#категориальные столбцы удаляем - это вместо методов уменьшения размерности 
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

#выбросы
sns.boxenplot(data=df_train)
plt.title('Выбросы')
plt.xticks(rotation=30)
plt.show()
#выбросов нет, смущает гигантский разрыв в численных данных всех столбцов со столбцом 'Policy Start Date'...
#пока оставим все, как есть

#перейдем к модели
X = df_train.drop('Premium Amount', axis=1)
y = df_train['Premium Amount']

#X_train, X_test, y_train, y_test = train_test_split(X.sample(frac=0.1, random_state=42), y.sample(frac=0.1, random_state=42), test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

dt_regressor = DecisionTreeRegressor(random_state=42)

grid_search = GridSearchCV(estimator=dt_regressor, param_grid=param_grid, 
                           scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("Лучшие гиперпараметры:", best_params)

best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
rmsle = np.sqrt(np.mean((np.log1p(y_test) - np.log1p(y_pred)) ** 2))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAPE: {mape}")
print(f"RMSLE: {rmsle}")
print(f"R^2: {r2}")