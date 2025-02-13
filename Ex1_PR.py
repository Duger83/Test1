import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures

print("Чтение данных...")

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
#более 1% пропусков обнаружено в 9-и столбцах, среди которых точно есть важные, например, Annual Income, Age ...
#отбросить все столбцы с пропусками нельзя + заполнять столько пропусков каким-нибудь средними нельзя - изменишь статистику
#в исходном файле более 1.2 миллиона строк - попробуем удалить пропуски

df_cleaned = df_train.dropna().drop('id', axis=1)
df_cleaned['Policy Start Date'] = pd.to_numeric(pd.to_datetime(df_cleaned['Policy Start Date']))
#после удаления пропусков осталось 384k строк, долждно хватить для обучения молелей, хоть это и всего 32% от исходных данных

#применим к категориальным столбцам метод OneHotEncoder: да, это вызовет значительное размерности матрицы по столбцам...
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

#на первом этапе вводил признак для богатых, но он в итоге коррелировал с доходом, что не удивительно,
#в итоге признак не вошел в итогвый набор для модели
#df_train['rich1'] = df_train['Annual Income'].map(lambda x: 1 if x>100000 else 0)
#sns.displot(data=df_train, x='Annual Income', hue='rich1')
#plt.show()

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
plt.show()
#стало лучше, но не идеально...
  
#матрица корреляций
correlation_matrix = df_train.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Исходная матрица парных корреляций')
plt.show()

#видно, что коррелирующих столбцов много, выпишем их в список drop_cols
drop_cols = ['Divorced', 'Employed', 'Rural', 'Basic', 'Average', 'Rarely', 'Condo', 'Monthly', "Bachelor's", "Master's", 'Credit Score']

#итоговая матрица корреляций 
correlation_matrix = df_train.drop(drop_cols, axis=1).corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Итоговая матрица парных корреляций')
plt.show()

corr_target = correlation_matrix['Premium Amount'].drop('Premium Amount')
selected_features = corr_target[abs(corr_target) > 0].index.tolist()

#нормализуем выбранные данные
X_min_max = df_train[selected_features]
for column in X_min_max.columns: 
	X_min_max[column] = (X_min_max[column] - X_min_max[column].min()) / (X_min_max[column].max() - X_min_max[column].min())	  

y_min_max = (df_train['Premium Amount'] - df_train['Premium Amount'].min()) / (df_train['Premium Amount'].max() - df_train['Premium Amount'].min())	   

X = X_min_max
y = y_min_max

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

#перейдем к полиномиальной модели
#попробуем заодно подобрать лучшую степень полинома
#четвертую степень мой Mac считал около 10 минут и нагрелся, пятую степень считать уже отказался, завершив процесс отладки
degrees = [1, 2, 3]  # поэтому пробуем степени 1, 2 и 3
mae_scores = []
mse_scores = []
r2_scores = []

#подбор степени
for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    X_poly_train = poly.fit_transform(X_train)
    model = LinearRegression()
    
    #кросс-валидация
    scores = cross_val_score(model, X_poly_train, y_train, cv=5, scoring='neg_mean_squared_error')
    mse_scores.append(-scores.mean())
    model.fit(X_poly_train, y_train)
    
    X_poly_test = poly.transform(X_test)
    y_pred = model.predict(X_poly_test)
    
    #оценка модели
    mae_scores.append(mean_absolute_error(y_test, y_pred))
    mse_scores.append(mean_squared_error(y_test, y_pred))
    r2_scores.append(r2_score(y_test, y_pred))

#результат
for i in range(3):
    print(f"Степень: {degrees[i]} -  MAE: {mae_scores[i]}, MSE: {mse_scores[i]}, R^2: {r2_scores[i]}")

#наилучшие показатели у квадратного полинома
#вообще MSE и MAE вполне хороши, но R^2 везде мизерный - для прогнозирования я не взял бы регрессию вовсе, 
#даже с полиномо второй степени