import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import joblib
from flaml import AutoML

print('Чтение данных...\n')

df=pd.read_csv('Data/hapi_original.csv')

cols = list(df.columns)

f = open('Results/nulls.csv', 'w')
print('Признак,Доля_пропусков', file = f) 

for col in cols:
    stats = [
        df[col].isnull().mean(),
    ]
    s = col + ','
    for i in stats:
        s = s + str(i) +','
    s = s[:-1]    
    print(s, file = f)
f.close()
print('Cоздан файл c вычислением доли пропусков по каждому признаку - Results/nulls.csv\n')

#в столбце alcohol_consumption обнаружено почти 60% (!) пропусков: 
#в этом столбце в дадасете есть значение None, которое трактуется как пропуск, однако, скорее всего это означает, 
#что человек не употребляет алкоголь и можно None заменить на No, если бы это был реальный пропуск данных в ячейке была бы просто пустая строка...
#исключать этот столбец не будем - заменим пропуск на текстовое No
df['alcohol_consumption'] = df['alcohol_consumption'].fillna('No')

#у нас 27 признаков, 10 из которых - категориальные
cat_cols = [
    'gender',
    'region',
    'income_level',
    'smoking_status',
    'alcohol_consumption',
    'physical_activity',
    'dietary_habits',
    'air_pollution_exposure',
    'stress_level',
    'EKG_results'
]
#причем в столбцах physical_activity, air_pollution_exposure и stress_level используются одинаковые значения категорий
#учтем это
df['physical_activity'] = df['physical_activity'].replace('High', 'High1')
df['physical_activity'] = df['physical_activity'].replace('Low', 'Low1')
df['physical_activity'] = df['physical_activity'].replace('Moderate', 'Moderate1')
df['air_pollution_exposure'] = df['air_pollution_exposure'].replace('High', 'High2')
df['air_pollution_exposure'] = df['air_pollution_exposure'].replace('Low', 'Low2')
df['air_pollution_exposure'] = df['air_pollution_exposure'].replace('Moderate', 'Moderate2')
df['stress_level'] = df['stress_level'].replace('High', 'High3')
df['stress_level'] = df['stress_level'].replace('Low', 'Low3')
df['stress_level'] = df['stress_level'].replace('Moderate', 'Moderate3')

#применим к категориальным столбцам метод OneHotEncoder и удаляем заведомо лишний столбец после кодирования
df_cleaned = df.copy()
one_hot = OneHotEncoder() 
for col in cat_cols:
    encoded = one_hot.fit_transform(df_cleaned[[col]])
    df_cleaned[one_hot.categories_[0]] = encoded.toarray()
    df_cleaned = df_cleaned.drop(col, axis=1).drop(one_hot.categories_[0][-1], axis=1)

df = df_cleaned.copy() #стало 33 столбца...

cols = list(df.columns)

f = open('Results/stats.csv', 'w')
print('Признак,Доля_пропусков,Максимум,Минимум,Среднее,Медиана,Дисперсия,Квантиль_0.1,Квантиль_0.9,Квартиль_1,Квартиль_3', file = f)

for col in cols:
    stats = [
        df[col].isnull().mean(),
        df[col].max(),
        df[col].min(),
        df[col].mean(),
        df[col].median(),
        df[col].var(),
        df[col].quantile(0.1),
        df[col].quantile(0.9),
        df[col].quantile(0.25),
        df[col].quantile(0.75)
    ]
    s = col + ','
    for i in stats:
        s = s + str(i) +','
    s = s[:-1]    
    print(s, file = f)
f.close()
print('Cоздан файл cо статичстическими показателями признаков - Results/stats.csv\n')

#построим матрицу парных корреляций
correlation_matrix = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title('Матрица парных корреляций')
plt.show()

#уберем из датафрейма по одному столбцу из пары признаков с корреляцией больше 0.2
threshold = 0.2
columns_to_drop = [
   'obesity', 
   'Low1', 
   'High3', 
   'Current',
   'High2', 
   'Moderate'
 ]
df_reduced = df.drop(columns=columns_to_drop)

#взглянем на матрицу корреляций еще раз
new_correlation_matrix = df_reduced.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(new_correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title('Матрица парных корреляций после удаления коррелирующих признаков')
plt.show()

df = df_reduced.copy() #осталось 26 признаков

#выбросы
for column in df.columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
sns.boxplot(data = df)
plt.title('Выбросы')
plt.xticks(rotation=30)
plt.show()
#выбросов данных нет
#нормировать данные не будем - разброс не очень велик...
df.to_csv('Data/hapi_prepared.csv')
print('Создан файл с подготовленными данными - Data/hapi_prepared.csv\n')

#модель KNN
X = df.drop(columns=['heart_attack'])
y = df['heart_attack']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier()

#подбор гиперпараметров
param_grid = {
    'n_neighbors': np.arange(10, 15),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
knn_cv = RandomizedSearchCV(knn, param_grid, n_iter=50, cv=5, random_state=42, n_jobs=-1)
knn_cv.fit(X_train, y_train)

#обучение модели с лучшими гиперпараметрами
best_knn = knn_cv.best_estimator_
y_pred = best_knn.predict(X_test)

#оценка модели
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, best_knn.predict_proba(X_test)[:, 1])

#ROC-кривая
fpr, tpr, thresholds = roc_curve(y_test, best_knn.predict_proba(X_test)[:, 1])
plt.figure()
plt.plot(fpr, tpr, label='ROC-кривая (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Ложноположительная ставка')
plt.ylabel('Истинноположительная ставка')
plt.title('ROC-кривая KNN')
plt.legend(loc="lower right")
plt.show()

#запись результатов в CSV
results = {
    'model': 'KNN',
    'best_params': str(knn_cv.best_params_),
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'roc_auc': roc_auc
}
results_df = pd.DataFrame([results])
results_df.to_csv('Results/models_results.csv', mode='a', index=False, header=True)

#модель SVC
X = df.drop(columns=['heart_attack'])
y = df['heart_attack']

#разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = SVC()

#подбор гиперпараметров
param_grid = {
    'C': [1, 10, 100],
    'kernel': ['rbf', 'poly', 'sigmoid']
}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)

#обучение модели с лучшими гиперпараметрами
best_model = grid_search.best_estimator_

#оценка модели
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.decision_function(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

#ROC-кривая
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure()
plt.plot(fpr, tpr, label='ROC-кривая (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Ложноположительная ставка')
plt.ylabel('Истинноположительная ставка')
plt.title('ROC-кривая SVC')
plt.legend(loc="lower right")
plt.show()

#запись результатов в CSV
results = {
    'model': 'SVC',
    'best_params': str(grid_search.best_params_),
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'roc_auc': roc_auc
}
results_df = pd.DataFrame([results])
results_df.to_csv('Results/models_results.csv', mode='a', index=False, header=False)

#модель LogisticRegression
X = df.drop('heart_attack', axis=1)
y = df['heart_attack'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression()

#подбор гиперпараметров
param_grid = {
    'C': [0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'max_iter': [100, 200, 300],
    'solver': ['liblinear', 'saga', 'lbfgs']
}

random_search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=5, random_state=42)
random_search.fit(X_train, y_train)

#обучение модели с лучшими гиперпараметрами
best_model = random_search.best_estimator_

y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

#оценка модели
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

#ROC-кривая
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure()
plt.plot(fpr, tpr, label='ROC-кривая (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Ложноположительная ставка')
plt.ylabel('Истинноположительная ставка')
plt.title('ROC-кривая LogisticRegression')
plt.legend(loc="lower right")
plt.show()

#запись результатов в CSV
results = {
    'model': 'LogisticRegression',
    'best_params': str(random_search.best_params_),
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'roc_auc': roc_auc
}
results_df = pd.DataFrame([results])
results_df.to_csv('Results/models_results.csv', mode='a', index=False, header=False)

#модель DecisionTree
X = df.drop(columns=['heart_attack'])
y = df['heart_attack']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#подбор гиперпараметров
param_grid = {
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

model = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

#обучение модели с лучшими гиперпараметрами
best_model = grid_search.best_estimator_

#оценка модели
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1] 

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

#ROC-кривая
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure()
plt.plot(fpr, tpr, label='ROC-кривая (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Ложноположительная ставка')
plt.ylabel('Истинноположительная ставка')
plt.title('ROC-кривая DecisionTree')
plt.legend(loc="lower right")
plt.show()

#запись результатов в CSV
results = {
    'model': 'DecisionTreeClassifier',
    'best_params': str(grid_search.best_params_),
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'roc_auc': roc_auc
}
results_df = pd.DataFrame([results])
results_df.to_csv('Results/models_results.csv', mode='a', index=False, header=False)

#модель RandomForest
X = df.drop('heart_attack', axis=1)
y = df['heart_attack']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(random_state=42)

#подбор гиперпараметров
param_grid = {
    'criterion' : ['gini', 'entropy', 'log_loss'],
    'n_estimators': [100, 200, 300],
    'max_depth': [30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=10, cv=3, random_state=42)
random_search.fit(X_train, y_train)

#оценка модели
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

#ROC-кривая
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, label='ROC-кривая (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Ложноположительная ставка')
plt.ylabel('Истинноположительная ставка')
plt.title('ROC-кривая RandomForest')
plt.legend(loc="lower right")
plt.show()

#запись результатов в CSV
results = {
    'model': 'RandomForestClassifier',
    'best_params': str(random_search.best_params_),
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'roc_auc': roc_auc
}
results_df = pd.DataFrame([results])
results_df.to_csv('Results/models_results.csv', mode='a', header=False, index=False)

print('Создан файл с данными о работе классификаторов - Results/models_results.csv\n')
#лучший результат показала модель RandomForest
#сохраним ее в файл
joblib.dump(best_model, 'Best_models/random_forest_model.pkl')
print('Сохранен файл с лучшей моделью - Best_models/random_forest_model.pkl\n')

#AutoML
X = df.drop(columns=['heart_attack'])
y = df['heart_attack']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#настройка FLAML
automl = AutoML()

automl_settings = {
    'time_budget': 120,  
    'metric': 'roc_auc',  
    'task': 'classification', 
    'log_file_name': 'automl.log', 
}

#обучение модели
automl.fit(X_train=X_train, y_train=y_train, **automl_settings)

#оценка модели
y_pred = automl.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, automl.predict_proba(X_test)[:, 1])

#ROC-кривая
fpr, tpr, _ = roc_curve(y_test, automl.predict_proba(X_test)[:, 1])
plt.figure()
plt.plot(fpr, tpr, label='ROC-кривая (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Ложноположительная ставка')
plt.ylabel('Истинноположительная ставка')
plt.title('ROC-кривая FLAML')
plt.legend(loc="lower right")
plt.show()

#запись результатов в CSV
results = {
    'model': 'AutoML',
    'best_params': ' ',
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'roc_auc': roc_auc
}
results_df = pd.DataFrame([results])
results_df.to_csv('models_results.csv', mode='a', index=False, header=False)

#сохранение модели
joblib.dump(automl, 'Best_models/FLAML_model.pkl')
print('Сохранен файл с автоматической моделью - Best_models/FLAML_model.pkl\n')

#формально лучший результат по метрике ROC_AUC показала автоматическая модель, но ее показатели чрезвычайно близки с показателями модели 
#RandomForestClassifier, у которой, кстати, выше показатель accuracy.