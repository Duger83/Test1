import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

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

#скользящее окно
df['rolling_mean'] = df['IPG2211A2N'].rolling(window=12).mean()
df['rolling_std'] = df['IPG2211A2N'].rolling(window=12).std()
plt.figure(figsize=(12, 6))
plt.plot(df['IPG2211A2N'], label='Исходные данные')
plt.plot(df['rolling_mean'], label='Скользящее среднее', color='red')
plt.plot(df['rolling_std'], label='Скользящее стандартное отклонение', color='orange')
plt.legend()
plt.show()

#сезонная декомпозиция
decomposition = seasonal_decompose(df['IPG2211A2N'], model='additive')
decomposition.plot()
plt.show()

#автокорреляция
plot_acf(df['IPG2211A2N'])
plt.show()

#частичная автокорреляция
plot_pacf(df['IPG2211A2N'])
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
model_auto = auto_arima(train['IPG2211A2N'], trace=True)
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
if mse_arima < mse_sarima:
    print('ARIMA показала лучшие результаты.')
else:
    print('SARIMA показала лучшие результаты.')