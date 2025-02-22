import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

print("Чтение данных...")

df_train = pd.read_csv('train_dataset.csv')
# категориальных столбцов нет

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
# пропусков в данных нет - это очень хорошо;
# средние значения данных по столбцам разнятся, скорее всего необходимо будет их нормировать или шкалировать
# скорее всего будут существенные корреляции между некоторыми столбцами - это видно и из названий и из беглого просмотра значений
 
print('Доля 1 (единиц) для целевой переменной smoking = ', df_train['smoking'].sum()/len(df_train))
# курильщиков < 37% - похоже на реальную ситуацию...

# что там с корреляцией столбцов?
correlation_matrix = df_train.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Исходная матрица парных корреляций')
plt.xticks(rotation=30)
plt.show()

#выбросы
sns.boxenplot(data=df_train)
plt.xticks(rotation=30)
plt.show()


# ниже были сделаны попытки избавиться от коррелирующих столбцов,
# убрать выбросы и шкалировать данные...

# попытка убрать коррелирующие столбцы или заменить их привела к ухудшению метрик модели!
"""
# как и ожидалось: корреляций много, уберем явные за счет усреднения
df_train['eyesight'] = (df_train['eyesight(left)'] + df_train['eyesight(right)']) / 2
df_train['hearing'] = (df_train['hearing(left)'] + df_train['hearing(right)']) / 2
df_train['pressure'] = (df_train['systolic'] + df_train['relaxation']) / 2
df_train['chol_ldl'] = (df_train['Cholesterol'] + df_train['LDL']) / 2
df_train['alt_ast_gtp'] = df_train['Gtp'] / ((df_train['ALT'] + df_train['AST']) / 2)

# для объединения роста-веса-окружности_талии-возраста воспользуемся модифицированной формулой Борнгарда
df_train['antro'] = (df_train['height(cm)'] * df_train['waist(cm)']) / (df_train['weight(kg)'] * 240 * df_train['age'])

# гемоглобин и свертываемость сильно коррелируют, но как их объединить - ?
# та же картина с показателями HDL и триглицерида...
# пока просто исключу свертываемость  и  HDL из анализа, т.к. от показателей гемоглобина и триглицерида целевая переменная зависит сильнее
# затем, возможно, попробую ввести в рассмотрение частное этих переменных

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
"""
 
df_train_new = df_train.copy()
# оценим выбросы значений и при необходимости обработаем их 
sns.boxenplot(data=df_train_new)
plt.xticks(rotation=30)
plt.show()

# выбросы есть и данные в разных шкалах
# обработаем выбросы и шкалируем данные

for col in df_train_new.drop(['smoking', 'dental caries'], axis=1).columns:
    Q3 = df_train_new[col].quantile(0.75)
    Q1 = df_train_new[col].quantile(0.25)
    IQR = Q3 - Q1
    upper = Q3 + 1.5 * IQR
    lower = Q1 - 1.5 * IQR
    df_train_new[col] = df_train_new[col].map(lambda x: pd.NA if x<lower else x if x<=upper else pd.NA)    
df_train_new = df_train_new.dropna()

df_train = df_train_new.copy()

scaler = StandardScaler()

X = pd.DataFrame(scaler.fit_transform(df_train.drop('smoking', axis=1)))
y = df_train['smoking'] 

sns.boxplot(data=X)
plt.xticks(rotation=30)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

param_grid = {
    'C': [1, 10, 100],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

grid_search = GridSearchCV(SVC(), param_grid, cv=4, scoring='roc_auc')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print('Лучшая модель - SVC(C =', best_model.C, ', kernel =', best_model.kernel, ')')

y_pred = best_model.predict(X_test)
y_pred_proba = best_model.decision_function(X_test)

print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))
print('F1 Score:', f1_score(y_test, y_pred))
print('ROC AUC:', roc_auc_score(y_test, y_pred_proba))

# результаты следующие:
# если убирать выбросы и шкалировать данные, а потом применять модель, 
# то лучший результат будет при С = 1 и kernel = rbf
# Accuracy: 0.7581047381546134
# ROC AUC: 0.8223797639448722
# 
# а если с данными вовсе ничего не делать, просто загрузить их в dataframe и применить модель,
# то лучший результат будет при С = 100 и kernel = rbf, причем результаты по метрикам будут
# почти идентичны приведенным выше

# разница в значенях параметра С как раз и объясняется отсутствием шкалирования данных,
# еще можно сказать, что, видимо, SVC устойчив к выбросам и независим от корреляций исходных данных