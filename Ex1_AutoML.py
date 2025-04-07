import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from flaml import AutoML
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import joblib

print("Чтение данных...\n")

df_train = pd.read_csv('train_dataset.csv')
#категориальных столбцов, как и ранее, нет

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
print('Cоздан файл со статистикой - stats.csv\n')
#пропусков в данных, как и ранее, нет

 
print('Доля 1 (единиц) для целевой переменной smoking = ', df_train['smoking'].sum()/len(df_train), '\n')
#курильщиков, как и ранее около 37%

correlation_matrix = df_train.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Исходная матрица парных корреляций')
plt.xticks(rotation=30)
plt.show()
#корелляций много...
#в этот раз все же попробуем поработать с данными

df_train['eyesight'] = (df_train['eyesight(left)'] + df_train['eyesight(right)']) / 2
df_train['hearing'] = (df_train['hearing(left)'] + df_train['hearing(right)']) / 2
df_train['pressure'] = (df_train['systolic'] + df_train['relaxation']) / 2
df_train['chol_ldl'] = (df_train['Cholesterol'] + df_train['LDL']) / 2
df_train['alt_ast_gtp'] = df_train['Gtp'] / ((df_train['ALT'] + df_train['AST']) / 2)

#для объединения роста-веса-окружности_талии-возраста воспользуемся модифицированной формулой Борнгарда
df_train['antro'] = (df_train['height(cm)'] * df_train['waist(cm)']) / (df_train['weight(kg)'] * 240 * df_train['age'])

#гемоглобин и свертываемость сильно коррелируют, 
#та же картина с показателями HDL и триглицерида...
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

#посмотрим на матрицу корреляций еще раз
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

#данные готовы, перейдем непосрелдственно к решению задачи

X = df_train_new.drop('smoking', axis=1)
y = df_train_new['smoking']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#cлучайное сэмплирование 
X_sample, _, y_sample, _ = train_test_split(X, y, test_size=0.8, random_state=42)

#настройка AutoML
automl = AutoML()
automl.fit(X_sample, y_sample, task='classification', time_budget=60)

#получение лучших моделей и их гиперпараметров
best_model = automl.best_estimator
best_params = automl.best_config
print("Лучшая модель:", best_model)
print("Ее гиперпараметры:", best_params, '\n')

y_pred = automl.predict(X_sample)
metrics = {
    'Accuracy': accuracy_score(y_sample, y_pred),
    'Precision': precision_score(y_sample, y_pred),
    'Recall': recall_score(y_sample, y_pred),
    'F1 Score': f1_score(y_sample, y_pred),
    'ROC AUC': roc_auc_score(y_sample, automl.predict_proba(X_sample)[:, 1])
}

metrics_df = pd.DataFrame(metrics, index=[0])
metrics_df.to_csv('Metrics.csv', mode='w', index=False, header=True)

#ROC-кривая
fpr, tpr, thresholds = roc_curve(y_sample, automl.predict_proba(X_sample)[:, 1])
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc_score(y_sample, automl.predict_proba(X_sample)[:, 1]))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Ложноположительная ставка')
plt.ylabel('Истинноположительная ставка')
plt.title('ROC-кривая FLAML AutoML')
plt.legend(loc='lower right')
plt.show()

#сохранение модели
joblib.dump(best_model, 'best_model.pkl')
#FLAML показал лучшие на этих данных результаты! Ранее Roc_AUC не поднимался выше 0.76, а тут - 0.96 и Accuracy - 0.91!