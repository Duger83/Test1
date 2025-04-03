import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from feature_engine.selection import DropFeatures
from feature_engine.creation import MathFeatures

print("Чтение данных...\n")

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
print('Cоздан файл c вычислением доли пропусков по каждому признаку - nulls.csv\n')
#более 1% пропусков обнаружено в 9-и столбцах, удаляем пропуски

drop_cols = ['id', 'Policy Start Date']
df_cleaned = df_train.dropna().drop(drop_cols, axis=1)
#df_cleaned['Policy Start Date'] = pd.to_numeric(pd.to_datetime(df_cleaned['Policy Start Date']))

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

#после кодирования категориальных столбцов убрал все появившиеся корреляции >= 0.3
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
print('Cоздан файл cо статичстическими показателями признаков - stats.csv\n')
  
#итоговая матрица корреляций
correlation_matrix = df_train.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Матрица парных корреляций')
plt.show()

#выбросы
sns.boxenplot(data=df_train)
plt.title('Выбросы')
plt.xticks(rotation=30)
plt.show()
#выбросов нет

#перейдем к модели
X = df_train.drop('Premium Amount', axis=1)
y = df_train['Premium Amount']

#случайная выборка данных для ускорения обучения модели
X_sample, _, y_sample, _ = train_test_split(X, y, train_size=0.2, random_state=42)

#разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.3, random_state=42)

#обучаем модель RandomForestRegressor
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

#предсказания на тестовой выборке
y_pred = model.predict(X_test)

#оценка качества модели
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
rmsle = np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_test))))
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae:.2f}')
print(f'MSE: {mse:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'MAPE: {mape:.2f}%')
print(f'RMSLE: {rmsle:.2f}')
print(f'R^2: {r2:.2f}')
print()

#вывод о качестве работы модели
if r2 < 0:
    print("Модель не объясняет вариацию целевой переменной.")
elif r2 < 0.5:
    print("Модель объясняет менее 50% вариации. Возможно, стоит улучшить модель.")
elif r2 < 0.8:
    print("Модель объясняет от 50% до 80% вариации. Результаты удовлетворительные.")
else:
    print("Модель хорошо объясняет вариацию целевой переменной.")
print()

#перейдем к генерации признаков

cols = [
        'Age', 
        'Annual Income', 
        'Number of Dependents', 
        'Health Score',
        'Previous Claims', 
        'Vehicle Age', 
        'Credit Score', 
        'Insurance Duration',
        'Female', 
        'Divorced', 
        "Bachelor's", 
        'Employed',
        'Suburban', 
        'Comprehensive', 
        'Average', 
        'No', 
        'Monthly', 
        'Apartment'
    ]

math_features = MathFeatures(variables=cols, func=['sum', 'mean', 'std']) 
df_features = math_features.fit_transform(df_train)

#сгенерированные столбцы для ознакомления
print('Сгенерированные признаки:')
print(df_features.columns)
print()

#фильтрация столбцов
df_features = df_features.dropna(axis=1)
threshold = 0.8 * len(df_features)  # 80% нулевых значений
df_features = df_features.loc[:, (df_features != 0).sum(axis=0) > threshold]
print('Оставшиеся признаки после фильтрации:')
print(df_features.columns)
print()

X = df_features.drop('Premium Amount', axis=1)
y = df_features['Premium Amount'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

#предсказания
y_pred = model.predict(X_test)

#оценка качества модели
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
rmsle = np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_test)) ** 2))
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae:.2f}')
print(f'MSE: {mse:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'MAPE: {mape:.2f}%')
print(f'RMSLE: {rmsle:.2f}')
print(f'R^2: {r2:.2f}')
print()

#важность признаков
importances = model.feature_importances_
feature_importances = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

print(feature_importances)

plt.figure(figsize=(12, 8))
plt.barh(feature_importances['Feature'], feature_importances['Importance'])
plt.xlabel('Важность')
plt.title('Важность признаков')
plt.show()
#значение метрики R^2 свалилось в множество отрицательных чисел после генерации признаков,
#поэтому модель формально ухудшилась... но она и так была неудовлетворительной для наших данных.