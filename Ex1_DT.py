import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
import graphviz

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

print("Чтение данных...")

df_train = pd.read_csv('train_dataset.csv')
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
#при решении задачи с SVM попытка убрать корелляции привела к ухудшению метрик модели,
#в этот раз все же попробуем поработать с ланными и не просто "выкинуть" кореллирующие признаки 

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

#выбросы
sns.boxenplot(data=df_train_new)
plt.xticks(rotation=30)
plt.show()

# критические выбросы есть в столбцах trigliceride и chol_idl (ну заодно все столбцы причешем, кроме целевого) - попробуем убрать их
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

# Подбор гиперпараметров
param_grid = {
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print("Подобранные значения гиперпараметров для лучшей модели:", best_params)

best_model = DecisionTreeClassifier(**best_params)
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
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

# Визуализация дерева решений c plot
plt.figure(figsize=(20, 10))
plot_tree(best_model, filled=True, feature_names=X.columns, class_names=['Non-Smoker', 'Smoker'])
plt.title("Decision Tree")
plt.savefig('Plots/decision_tree.png')  # Сохранение визуализации
plt.show()

# Определение важности признаков
feature_importances = best_model.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Выбор столбца с наибольшим весом
most_important_feature = importance_df.iloc[0]
print("Наиболее важный признак:", most_important_feature['Feature'], "с важностью:", most_important_feature['Importance'])
#как и ожидалось (по матрице корреляций), был выбран столбец 'hemoglobin'

# Модель с одним столбцом
X_single = df_train_new[['hemoglobin']]
y_single = df_train_new['smoking']

X_train_single, X_test_single, y_train_single, y_test_single = train_test_split(X_single, y_single, test_size=0.3, random_state=42)

single_model = DecisionTreeClassifier(**best_params)
single_model.fit(X_train_single, y_train_single)

y_pred_single = single_model.predict(X_test_single)
y_pred_proba_single = single_model.predict_proba(X_test_single)[:, 1]

conf_matrix_single = confusion_matrix(y_test_single, y_pred_single)
accuracy_single = accuracy_score(y_test_single, y_pred_single)
precision_single = precision_score(y_test_single, y_pred_single)
recall_single = recall_score(y_test_single, y_pred_single)
f1_single = f1_score(y_test_single, y_pred_single)
roc_auc_single = roc_auc_score(y_test_single, y_pred_proba_single)

print("Качество модели с одним признаком 'hemoglobin':")
print("Confusion Matrix:\n", conf_matrix_single)
print("Accuracy:", accuracy_single)
print("Precision:", precision_single)
print("Recall:", recall_single)
print("F1 Score:", f1_single)
print("ROC AUC:", roc_auc_single)

# Визуализация дерева решений c graphviz
dot_data_single = export_graphviz(single_model, out_file=None, 
                                   feature_names=X_single.columns,  
                                   class_names=['Non-Smoker', 'Smoker'],  
                                   filled=True, rounded=True,  
                                   special_characters=True)  
graph_single = graphviz.Source(dot_data_single)  
graph_single.render("decision_tree_single") 