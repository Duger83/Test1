import os
import pandas as pd
from pyspark.sql import SparkSession 
spark = SparkSession.builder.master("local[*]").getOrCreate()
spark.conf.set("spark.sql.repl.eagerEval.enabled", True) 

print("Объединение таблиц:")
df_l1 = spark.read.csv('tables_csv/actor.csv', header=True, sep=",").drop("last_update")
df_l2 = spark.read.csv('tables_csv/film_actor.csv', header=True, sep=",").drop("last_update")
df_l1 = df_l1.join(df_l2, ["actor_id"])

df_l2 = spark.read.csv('tables_csv/film.csv', header=True, sep=",").drop("last_update", "original_language_id")
df_l1 = df_l1.join(df_l2, ["film_id"])

df_l2 = spark.read.csv('tables_csv/film_category.csv', header=True, sep=",").drop("last_update")
df_l1 = df_l1.join(df_l2, ["film_id"])

df_l2 = spark.read.csv('tables_csv/category.csv', header=True, sep=",").drop("last_update")
df_l1 = df_l1.join(df_l2, ["category_id"])

df_l1.write.csv('results/3', mode='overwrite', header=True)
files = os.listdir('results/3')
for i in files:
    n, e = os.path.splitext(i)
    if e == '.csv':
        os.rename('results/3/'+i,  'results/3/super_table.csv')

print('Итоговая таблица super_table создана в папке results/3_3')

df = pd.read_csv("results/3/super_table.csv")
f = open('results/3/stats.csv', 'w')
print('Признак,Доля_пропусков,Максимум,Минимум,Среднее,Медиана,Дисперсия,Квантиль_0.1,Квантиль_0.9,Квартиль_1,Квартиль_3,Кол-во_уникальных,Мода', file = f)

numeric_cols = [
    'film_id', 
    'actor_id', 
    'release_year', 
    'language_id', 
    'rental_duration',
    'rental_rate',
    'length',
    'replacement_cost'
]

categorial_cols = [
    'first_name', 
    'last_name', 
    'description', 
    'name', 
    'title',
    'rating',
    'special_features',
]

for col in numeric_cols:
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
    s = s + ','    
    print(s, file = f)

for col in categorial_cols:
    stats = [
        df[col].isnull().mean(),
        df[col].nunique(),
        df[col].mode()[0] 
    ]
    if col == 'name':
        s = 'category,' + str(stats[0]) + ',,,,,,,,,,' + str(stats[1]) + ',' + str(stats[2])
    else:
        if col != 'special_features':
            s = col + ',' + str(stats[0]) + ',,,,,,,,,,' + str(stats[1]) + ',' + str(stats[2])
        else:
            s = col + ',' + str(stats[0]) + ',,,,,,,,,,' + str(stats[1]) + ',"' + str(stats[2]) + '"'
    print(s, file = f)

f.close()
print('Cоздан файл со статистикой - stats.csv')