import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder

print("Чтение данных...")
print()

df_train = pd.read_csv('train.csv')

cols = list(df_train.drop('id', axis=1).columns)

f = open('nulls.csv', 'w')
print('Признак,Доля_пропусков', file = f) 
      #,Максимум,Минимум,Среднее,Медиана,Дисперсия,Квантиль_0.1,Квантиль_0.9,Квартиль_1,Квартиль_3', file = f)

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
#более 1% пропусков обнаружено в 9-и столбцах, среди которых точно есть важные, например, Age
#отбросить все столбцы с пропусками нельзя + заполнять столько пропусков каким-нибудь средними нельзя - изменишь статистику
#в исходном файле более 1.2 миллиона строк - попробуем удалить пропуски

df_cleaned = df_train.dropna().drop('id', axis=1)
df_cleaned['Policy Start Date'] = pd.to_numeric(pd.to_datetime(df_cleaned['Policy Start Date']))
#после удаления пропусков осталось 384k строк, долждно хватить для обучения молелей, хоть это и всего 32% от исходных данных

#применим к категориальным столбцам метод OneHotEncoder: да, это вызовет значительное увеличение размерности матрицы по столбцам...
#но, может быть, в дальнейшем можно будет исключить некоторые столбцы после проверки парных корреляций
#также удаляем заведомо лишний столбец после кодирования
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

df_train = df_cleaned.copy() #стало 30 столбцов

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
    #sns.boxplot(data = df_train[col])
    #plt.show()
f.close()
print('Cоздан файл cо статичстическими показателями признаков - stats.csv')
print()

#в Premium Amount есть выбросы, спровоцированные богачами, 
# попробуем их убрать
Q3 = df_train['Premium Amount'].quantile(0.75)
Q1 = df_train['Premium Amount'].quantile(0.25)
IQR = Q3 - Q1
upper = Q3 + 1.5 * IQR
lower = Q1 - 1.5 * IQR
df_train = df_train[(df_train['Premium Amount']<upper)&(df_train['Premium Amount']>=lower)]
df_train['Premium Amount'] = df_train['Premium Amount'].map(lambda x: pd.NA if x<lower else x if x<=upper else pd.NA)
sns.boxplot(data = df_train['Premium Amount'])
#plt.show()
#стало лучше, но не идеально...
  
#матрица корреляций
correlation_matrix = df_train.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Исходная матрица парных корреляций')
#plt.show()

#видно, что коррелирующих столбцов много, выпишем их в список drop_cols
drop_cols = ['Divorced', 'Employed', 'Rural', 'Basic', 'Average', 'Rarely', 'Condo', 'Monthly', "Bachelor's", "Master's", 'Credit Score']

#итоговая матрица корреляций 
correlation_matrix = df_train.drop(drop_cols, axis=1).corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Итоговая матрица парных корреляций')
#plt.show()

corr_target = correlation_matrix['Premium Amount'].drop('Premium Amount')
selected_features = corr_target[abs(corr_target) >= 0].index.tolist()

#нормализуем выбранные данные
pd.options.mode.chained_assignment = None
X_min_max = df_train[selected_features]
for column in X_min_max.columns: 
	X_min_max[column] = (X_min_max[column] - X_min_max[column].min()) / (X_min_max[column].max() - X_min_max[column].min())	  

y_min_max = (df_train['Premium Amount'] - df_train['Premium Amount'].min()) / (df_train['Premium Amount'].max() - df_train['Premium Amount'].min())	   

X = X_min_max
y = y_min_max


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15)

#перейдем к моделям
models = {
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'ElasticNet': ElasticNet()
}

results = {}

# обучение и оценка 
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MAE': mae, 'MSE': mse, 'R^2': r2}

best_model_name = max(results, key=lambda x: results[x]['R^2'])
best_model = results[best_model_name]

print('Результаты применения моделей к данным:')
for name, metrics in results.items():
    print(f"{name}: MAE = {metrics['MAE']}, MSE = {metrics['MSE']}, R^2 = {metrics['R^2']}")

print()
print(f'Лучшая модель до подбора гиперпараметров: {best_model_name} с R^2 = {best_model['R^2']}')

#по метрике R^2 все модели ОЧЕНЬ далеки от идеала, но у Ridge этот показатель хотя бы > 0
 
#посмотрим, что из выйдет после подбора гиперапараметров по сетке
param_grid = {
    'Ridge': {
        'alpha': [0.1, 1.0, 10, 20, 30],
        'fit_intercept': [True, False],
        'positive': [True, False],
        'max_iter': [1000, 2000],
        'random_state' : [2, 5, 10]
    },
    'Lasso': {
        'alpha': [0.0001, 0.001, 0.1, 1.0, 10],
        'fit_intercept': [True, False],
        'positive': [True, False],
        'precompute': [True, False],
        'max_iter': [1000, 2000],
        'random_state' : [2, 5, 10]
    },
    'ElasticNet': {
        'alpha': [0.0001, 0.001, 0.1, 1.0, 10],
        'l1_ratio': [0.1, 0.5, 0.9],
        'fit_intercept': [True, False],
        'positive': [True, False],
        'precompute': [True, False],
        'max_iter': [1000, 2000],
        'random_state' : [2, 5, 10]
    }
}

print()
best_models = {}
for model_name, model in models.items():
    grid_search = GridSearchCV(model, param_grid[model_name], scoring='neg_mean_squared_error', cv=2)
    grid_search.fit(X_train, y_train)
    best_models[model_name] = grid_search.best_estimator_
    print(f'Лучшие гиперпараметры для {model_name}: {grid_search.best_params_}')

print()
print('Результаты применения моделей к данным после подбора гиперпараметров:')
for model_name, model in best_models.items():
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'{model_name} : MAE = {mae}, MSE = {mse}, R^2 = {r2}')

best_model_name = max(best_models.keys(), key=lambda x: r2_score(y_test, best_models[x].predict(X_test)))
print()
print(f'Лучшая модель после подбора гиперпараметров: {best_model_name} с R^2 = {best_model['R^2']}')
#подбор гиперпараметров улучшил знасчение метрики R^2 для моделей Lasso и ElasticNet,
#но эти значения по-прежнему очень далеки от идеала...