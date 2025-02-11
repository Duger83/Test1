import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, f1_score

print("Чтение данных...")

df_train = pd.read_csv('train.csv').drop('id', axis=1)

numeric_cols = list(df_train.columns)

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

sns.scatterplot(data = df_train, x = 'ph', y = 'osmo')
plt.ylabel('OSMO')
plt.xlabel('Ph')
plt.show()

sns.scatterplot(data = df_train, x = 'urea', y = 'gravity')
plt.ylabel('Gravity')
plt.xlabel('Urea')
plt.show()

X_min_max = df_train[['gravity', 'ph', 'osmo', 'urea', 'calc']]

for column in X_min_max.columns: 
	X_min_max[column] = (X_min_max[column] - X_min_max[column].min()) / (X_min_max[column].max() - X_min_max[column].min())

X = X_min_max
Y = df_train['target']

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.3, random_state=10)

param_grid = {'n_neighbors': range(1, 21)}
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5)
grid_search.fit(X_train, Y_train)

print('Доля 1 (единиц) для целевой переменной target = ', df_train['target'].sum()/len(df_train))

best_k = grid_search.best_params_['n_neighbors']
print(f'Оптимальное значение k: {best_k}')

model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, Y_train)

Y_pred = model.predict(X_val)

conf_matrix = confusion_matrix(Y_val, Y_pred)
accuracy = accuracy_score(Y_val, Y_pred)
precision = precision_score(Y_val, Y_pred)
recall = recall_score(Y_val, Y_pred)
f1 = f1_score(Y_val, Y_pred)
roc_auc = roc_auc_score(Y_val, model.predict_proba(X_val)[:, 1])

print('Confusion matrix:')
print(conf_matrix)
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 score: {f1:.2f}')
print(f'ROC AUC: {roc_auc:.2f}')

df_test = pd.read_csv('test.csv')
X_test = df_test[['gravity', 'ph', 'osmo', 'urea', 'calc']]  
Y_test_pred = model.predict(X_test)
df_test['target'] = Y_test_pred  
df_test.to_csv('test_pred.csv', sep=',', index=False, encoding='utf-8')

print('Создан файл с предсказаниями целевой переменной terget - test_pred.csv')