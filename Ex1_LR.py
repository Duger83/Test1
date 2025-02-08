import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

print("Чтение данных...")

df_train = pd.read_csv('train.csv')

numeric_cols = list(df_train.columns)

f = open('stats.csv', 'w')
print('Признак,Доля_пропусков,Максимум,Минимум,Среднее,Медиана,Дисперсия,Квантиль_0.1,Квантиль_0.9,Квартиль_1,Квартиль_3', file = f)

for col in numeric_cols:
    if col != 'id':
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
        sns.boxplot(data = df_train, y = col)
        plt.ylabel(col)
        plt.show()
f.close()
print('Cоздан файл со статистикой - stats.csv')
#в файле видно, что никаких пропусков нет

#на графике видно, что выбросов в значениях целевой переменной нет
#также на графиках видно, что предположительно выбросы есть в столбцах: num_children_at_home, store_sales(in millions), unit_sales(in millions)
#в анализе статистики видно, что значения столбца num_children_at_home нельзя считать выбросами 
#удаляем выбросы
cols = ['store_sales(in millions)', 'unit_sales(in millions)']
for i in cols:
    Q3 = df_train[i].quantile(0.75)
    Q1 = df_train[i].quantile(0.25)
    IQR = Q3 - Q1
    upper = Q3 + 1.5 * IQR
    lower = Q1 - 1.5 * IQR
    df_train[i+'_without_outliers'] = df_train[i].map(lambda x: np.NaN if x<lower else x if x<=upper else np.NaN)
    sns.boxplot(data = df_train, y = i+'_without_outliers')
    plt.ylabel(i+'_without_outliers')
    plt.show()
    
#матрица корреляций
correlation_matrix = df_train.drop('id', axis = 1).drop(cols, axis = 1).corr()

sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Матрица парных корреляций')
plt.show()

corr_target = correlation_matrix['cost'].drop('cost')

#входные переменные (... хотя глядя на матрицу, я бы сказал, что cost ни с одной переменной, похоже, вообще не коррелирует...)
selected_features = corr_target[abs(corr_target) > 0.04].index.tolist()
print('Выбранные независимые переменные для модели линейной регрессии:')
print(selected_features)

X = df_train[selected_features]
y = df_train['cost']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_sm = sm.add_constant(X_train)  
ols_model = sm.OLS(y_train, X_train_sm).fit()

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

coeff = pd.DataFrame(lr_model.coef_, X.columns, columns=['coeff'])
print(coeff)

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
if(r2_ols < 0.5) and (r2_lr < 0.5):
     print('Вывод - модели никуда не годятся. Точнее, никуда не годятся данные! Это было понятно еще по матрице корреляций.')
     print('В этой связи прогнозировать значение cost в фале test считаю бессмысленным.')
else:
    print('Модели хороши!')
    df_test = pd.read_csv('test.csv')
    X_test_final = df_test[selected_features]
    y_test_pred = lr_model.predict(X_test_final)
    df_test['pred_cost'] = y_test_pred
    df_test.to_csv('test_pred_cost.csv', index=True)    