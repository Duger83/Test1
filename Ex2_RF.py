import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.preprocessing import OneHotEncoder

print("Чтение данных...")

df_train = pd.read_csv('regression/train.csv')

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


#категориальные столбцы закодируем
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

one_hot = OneHotEncoder() 
for col in cat_cols:
    encoded = one_hot.fit_transform(df_cleaned[[col]])
    df_cleaned[one_hot.categories_[0]] = encoded.toarray()
    df_cleaned = df_cleaned.drop(col, axis=1).drop(one_hot.categories_[0][-1], axis=1)

cor_cols = ['Married',
            'High School',
            "Master's",
            'Self-Employed',
            'Rural',
            'Good',
            'Rarely',
            'Condo',
            'Basic',
            'Daily'
]

df_train = df_cleaned.drop(cor_cols, axis=1).copy()

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
#после кодирования категориальных столбцов убрал все появившиеся корреляции >= 0.3

#выбросы
sns.boxenplot(data=df_train)
plt.title('Выбросы')
plt.xticks(rotation=30)
plt.show()
#выбросов нет

#перейдем к модели
X = df_train.drop('Premium Amount', axis=1)
y = df_train['Premium Amount']

#случайная выборка данных для ускорения подбора гиперпараметров
X_sample, _, y_sample, _ = train_test_split(X, y, train_size=0.1, random_state=42)

#разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.3, random_state=42)

#подбор гиперпараметров
param_grid = {
    'n_estimators': [300, 200],
    'max_depth': [10, 20],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4],
    'max_features': ['sqrt', 'log2'],
   #'criterion': ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'] #вызывает неоправданную задержку даже на случайной малой выборке
}

rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
print("Лучшие гиперпараметры:", grid_search.best_params_)

#обучение модели
best_rf = grid_search.best_estimator_
best_rf.fit(X_train, y_train)

#прогнозирование на тестовой выборке
y_pred = best_rf.predict(X_test)

#оценка качества модели
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
rmsle = np.sqrt(np.mean((np.log1p(y_test) - np.log1p(y_pred)) ** 2))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"RMSLE: {rmsle:.2f}")
print(f"R^2: {r2:.2f}")

#поскольку в некоторых столбцах данные количественно отличаются во много раз и при этом не нормируются, 
#значения метрик MAE/MSE будут не очень показательны
#подготовка данных и подбор гиперпараметров модели позволи получить метрику RMSLE = 1,15 и метрику R^2 = 0,03
#это заметно лучше, чем при использовании SVR или KNN, и почти столько же, как при использовании Decision Tree, 
#но все равно недостаточно для хороших реальных прогнозов...