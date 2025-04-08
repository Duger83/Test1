import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from flaml import AutoML
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

#чтение данных
df = pd.read_csv('CSV/electric-production.csv')

#пропуски
missing_values = df.isnull().sum()

#установка 'DATE' в качестве индекса
df['DATE'] = pd.to_datetime(df['DATE'])
df.set_index('DATE', inplace=True)

#проверка индекса
is_date_index = isinstance(df.index, pd.DatetimeIndex)
answ = 'Нет'
if is_date_index:
    answ = 'Да'
print(f"'DATE' является индексом? {answ}\n")

#EDA
print(df.describe())
print('miss:')
print(missing_values, '\n')
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x=df.index, y='IPG2211A2N')
plt.title('Выработка элетктричества')
plt.xlabel('Дата')
plt.ylabel('Производство')
plt.show()

#стационарность
result = adfuller(df['IPG2211A2N'])
print('Статистика теста ADF:', result[0])
print('p-значение:', result[1])
print('Критические значения:')
for key, value in result[4].items():
    print(f'  {key}: {value}')

if result[1] <= 0.05:
    print('Ряд стационарен\n')
else:
    print('Ряд не стационарен\n')

#разделяем данные
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

print(f'Размер тренировочной выборки: {len(train)}')
print(f'Размер тестовой выборки: {len(test)}\n')

#модель ARIMA
model_auto = auto_arima(train['IPG2211A2N'], seasonal=True, trace=True)
print(model_auto.summary(), '\n')

p, d, q = model_auto.order
model_arima = ARIMA(train['IPG2211A2N'], seasonal_order=(p, d, q, 12))
model_arima_fit = model_arima.fit()

#прогноз
forecast_arima = model_arima_fit.forecast(steps=len(test))
mse_arima = mean_squared_error(test['IPG2211A2N'], forecast_arima)
plt.figure(figsize=(12, 6))
plt.plot(train['IPG2211A2N'], label='Тренировочная выборка')
plt.plot(test['IPG2211A2N'], label='Тестовая выборка')
plt.plot(test.index, forecast_arima, label='Прогноз ARIMA', color='red')
plt.legend()
plt.show()

#модель SARIMA
model_auto_sarima = auto_arima(train['IPG2211A2N'], seasonal=True, m=12, trace=True)
print(model_auto_sarima.summary(), '\n')

P, D, Q, m = model_auto_sarima.order + (model_auto_sarima.seasonal_order[0],)
model_sarima = SARIMAX(train['IPG2211A2N'], order=(p, d, q), seasonal_order=(P, D, Q, 12))

model_sarima_fit = model_sarima.fit()

#прогноз
forecast_sarima = model_sarima_fit.forecast(steps=len(test))
mse_sarima = mean_squared_error(test['IPG2211A2N'], forecast_sarima)
plt.figure(figsize=(12, 6))
plt.plot(train['IPG2211A2N'], label='Тренировочная выборка')
plt.plot(test['IPG2211A2N'], label='Тестовая выборка')
plt.plot(test.index, forecast_sarima, label='Прогноз SARIMA', color='green')
plt.legend()
plt.show()

#итог
print('\nВыводы:')
print(f'MSE модели ARIMA: {mse_arima}')
print(f'MSE модели SARIMA: {mse_sarima}')


#AutoML
df = pd.read_csv('CSV/electric-production.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
#df.set_index('DATE', inplace=True)

train_size = int(len(df) * 0.7)
train, test = df[:train_size], df[train_size:]

X_train = train[['DATE']]
y_train = train['IPG2211A2N']
X_test = test[['DATE']]
y_test = test['IPG2211A2N']

automl = AutoML('mse')
automl.fit(X_train, y_train, time_budget=180,  task="ts_forecast", period=12)

#оценка
y_pred = automl.predict(X_test)

plt.figure(figsize=(12, 6))
plt.plot(train['DATE'], y_train, label='Тренировочная выборка', color='blue')
plt.plot(test['DATE'], y_test, label='Тестовая выборка', color='orange')
plt.plot(test['DATE'], y_pred, label='Прогноз AutoML', color='green')
plt.xlabel('Дата')
plt.ylabel('Выработка')
plt.title('Прогноз AutoML')
plt.legend()
plt.show()

best_model = automl.best_estimator
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Метрики для ', best_model)
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"R^2: {r2:.4f}")

joblib.dump(best_model, 'best_model.pkl')

print("Лучшая модель сохранена в файл best_model.pkl")
#лучшая модель AutoML более чем в два раза превзошла модели, обученные вручную