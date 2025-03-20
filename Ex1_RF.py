import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

print("Чтение данных...")

df_train = pd.read_csv('smoking_data/train_dataset.csv')
# категориальных столбцов, как и ранее, нет

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
print('Cоздан файл со статистикой - stats.csv')
# пропусков в данных, как и ранее, нет

 
print('Доля 1 (единиц) для целевой переменной smoking = ', df_train['smoking'].sum()/len(df_train))
# курильщиков, как и ранее около 37%

correlation_matrix = df_train.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Исходная матрица парных корреляций')
plt.xticks(rotation=30)
plt.show()
#корелляций много...

#поработаем с данными, не "выкидывая" кореллирующие признаки 
df_train['eyesight'] = (df_train['eyesight(left)'] + df_train['eyesight(right)']) / 2
df_train['hearing'] = (df_train['hearing(left)'] + df_train['hearing(right)']) / 2
df_train['pressure'] = (df_train['systolic'] + df_train['relaxation']) / 2
df_train['chol_ldl'] = (df_train['Cholesterol'] + df_train['LDL']) / 2
df_train['alt_ast_gtp'] = df_train['Gtp'] / ((df_train['ALT'] + df_train['AST']) / 2)

# для объединения роста-веса-окружности_талии-возраста воспользуемся модифицированной формулой Борнгарда
df_train['antro'] = (df_train['height(cm)'] * df_train['waist(cm)']) / (df_train['weight(kg)'] * 240 * df_train['age'])

#исключим свертываемость  и  HDL из анализа, т.к. от показателей гемоглобина и триглицерида целевая переменная зависит сильнее
drop_cols = ['age',
             'eyesight(left)', 
             'eyesight(right)', 
             'hearing(left)',  
             'hearing(right)', 
             'systolic', 
             'relaxation',
             'Cholesterol',
             'LDL',
             'ALT',
             'AST',
             'Gtp',
             'height(cm)',
             'waist(cm)',
             'weight(kg)',
             'serum creatinine',
             'HDL'
            ]

# посмотрим на матрицу корреляций еще раз
df_train_new = df_train.drop(drop_cols, axis=1).copy()
correlation_matrix = df_train_new.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Матрица парных корреляций в первом приближении')
plt.xticks(rotation=30)
plt.show()

#выбросы
sns.boxenplot(data=df_train_new)
plt.xticks(rotation=30)
plt.show()

for col in df_train_new.drop(['smoking'], axis=1).columns:
    Q3 = df_train_new[col].quantile(0.75)
    Q1 = df_train_new[col].quantile(0.25)
    IQR = Q3 - Q1
    upper = Q3 + 1.5 * IQR
    lower = Q1 - 1.5 * IQR
    df_train_new[col] = df_train_new[col].map(lambda x: pd.NA if x<lower else x if x<=upper else pd.NA) 
df_train_new = df_train_new.dropna()  
sns.boxenplot(data=df_train_new)
plt.xticks(rotation=30)
plt.show() 

#данные готовы, перейдем непосредственно к решению задачи
X = df_train_new.drop('smoking', axis=1)
y = df_train_new['smoking']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

#гиперпараметры
param_grid = {
    'criterion' : ['gini', 'entropy', 'log_loss'],
    'n_estimators': [100, 200, 300],
    'max_depth': [30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)
print("Лучшие гиперпараметры:", grid_search.best_params_)

#обучаем модель 
best_rf = grid_search.best_estimator_

#прогнозируем
y_pred = best_rf.predict(X_test)
y_pred_proba = best_rf.predict_proba(X_test)[:, 1]

#оценка качества модели
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
print("Confusion Matrix:\n", conf_matrix)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC:", roc_auc)
#значение метрик Accuracy=0.8 и ROC AUC=0.87 говорят о том, что модель предсказывает значение целевой перемнной достаточно хорошо
#похоже, что это первая модель, которая показала такие результаты