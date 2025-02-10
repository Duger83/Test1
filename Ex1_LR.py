import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

print("Чтение данных...")

df_train = pd.read_csv('train.csv')

numeric_cols = list(df_train.drop('id', axis=1).columns)

f = open('stats.csv', 'w')
print('Признак,Доля_пропусков,Максимум,Минимум,Среднее,Медиана,Дисперсия,Квантиль_0.1,Квантиль_0.9,Квартиль_1,Квартиль_3', file = f)

for col in numeric_cols:
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
#в файле видно, что нигде никаких пропусков нет

sns.boxplot(data = df_train['cost'])
plt.show()
#на графике видно, что критических выбросов в значениях целевой переменной нет
  
#матрица корреляций
correlation_matrix = df_train.drop('id', axis=1).corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Исходная матрица парных корреляций')
plt.show()

#видно, что коррелируют друг с другом 'store_sales(in millions)' и 'unit_sales(in millions)' - заменим их частным sales
df_train['sales'] = (df_train['store_sales(in millions)'] / df_train['unit_sales(in millions)'])

#видно, что коррелируют друг с другом 'total_children' и 'num_children_at_home' - заменим их средней суммой children
df_train['children'] = (df_train['total_children'] + df_train['num_children_at_home']) / 2

#также коорелирует пятерка столбцов 'coffee_bar', 'video_store', 'salad_bar', 'prepeared_food' и 'florist' - навскидку заменим их средней суммой 'stores',
#хотя не известно, на сколько это корректно...
df_train['stores'] = (df_train['video_store'] + df_train['salad_bar'] + df_train['prepared_food'] + df_train['florist'] + df_train['coffee_bar']) / 5

drop_cols = [
    'id', 
    'store_sales(in millions)', 
    'unit_sales(in millions)', 
    'total_children', 
    'num_children_at_home', 
    'video_store', 
    'salad_bar', 
    'prepared_food', 
    'coffee_bar',
    'florist'
    ]
correlation_matrix = df_train.drop(drop_cols, axis=1).corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Матрица парных корреляций после обработки')
plt.show()

corr_target = correlation_matrix['cost'].drop('cost')
selected_features = corr_target[abs(corr_target) > 0].index.tolist()

#работа с полученными данными
X = df_train[selected_features]
y = df_train['cost']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

X_train_sm = sm.add_constant(X_train)  
ols_model = sm.OLS(y_train, X_train_sm).fit()

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

coeff = pd.DataFrame(lr_model.coef_, X.columns, columns=['coeff'])

sns.barplot(x=coeff['coeff'], y=coeff.index)
plt.title('Веса признаков модели Linear Regression')
plt.xlabel('Значение')
plt.ylabel('Признак')
plt.show()

# Оценка моделей
y_pred_ols = ols_model.predict(X_train_sm)
y_pred_lr = lr_model.predict(X_test)

mse_ols = mean_squared_error(y_train, y_pred_ols)
r2_ols = r2_score(y_train, y_pred_ols)

mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print('OLS Model - MSE : ', mse_ols, 'R^2 : ', r2_ols)
print('Linear Regression Model - MSE : ', mse_lr, 'R^2 : ', r2_lr)
#огромныве значения MSE и мизерные значения R^2 говорят о том, что применяемые модели не подходят для имеющщегося набора данных...
#либо я что-то не так делаю, что намного вероятнее) 