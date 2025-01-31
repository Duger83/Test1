import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession 
from pyspark.sql.functions import concat_ws, col

spark = SparkSession.builder.master("local[*]").getOrCreate()
spark.conf.set("spark.sql.repl.eagerEval.enabled", True) 

def my_file_making(number, s):
    spark.sql(s).write.csv('results/2_'+str(number), mode='overwrite', header=True)
    files = os.listdir('results/2_'+str(number))
    for i in files:
        n, e = os.path.splitext(i)
        if e == '.csv':
            os.rename('results/2_'+str(number)+'/'+i,  'results/2_'+str(number)+'/res_2_' + str(number) + '.csv')
    

print('Формирование файлов с результатами запросов...')

#первый запрос
df_films = spark.read.csv('tables_csv/film.csv', header=True, sep=",")
df_films.createOrReplaceTempView("temp1")
s = """
    SELECT 
        rating,
        COUNT(*) AS total_films,
        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () AS percentage
    FROM 
        temp1
    GROUP BY 
        rating
    ORDER BY 
        rating
 """       
my_file_making(1, s)

#второй запрос
df_rental = spark.read.csv('tables_csv/rental.csv', header=True, sep=",").drop("customer_id", "rental_date", "last_update", "staff_id", "return_date")
df_inventory = spark.read.csv('tables_csv/inventory.csv', header=True, sep=",").drop("last_update", "store_id")
df_res = df_rental.join(df_inventory, ["inventory_id"])
df_film = spark.read.csv('tables_csv/film.csv', header=True, sep=",").drop("title", "description", "release_year", "last_update", "language_id", "original_language_id", "rental_duration", "length", "rental_rate", "replacement_cost", "special_features")
df_res = df_res.join(df_film, ["film_id"])
df_res.createOrReplaceTempView("temp1")
s = """
    SELECT 
        COUNT(rating) as count, 
        rating 
    FROM 
        temp1 
    GROUP BY 
        rating 
    ORDER BY 
        rating
"""
my_file_making(2, s)

#третий запрос
df_category = spark.read.csv('tables_csv/category.csv', header=True, sep=",").drop("last_update")
df_film_category = spark.read.csv('tables_csv/film_category.csv', header=True, sep=",").drop("last_update")
df_res = df_category.join(df_film_category, ["category_id"])
df_film = spark.read.csv('tables_csv/film.csv', header=True, sep=",").drop("title", "description", "release_year", "last_update", "language_id", "original_language_id", "length", "rental_rate", "replacement_cost", "special_features",  "rating")
df_res = df_res.join(df_film, ["film_id"])
df_res.createOrReplaceTempView("temp1")
s = """
    SELECT 
        name,
        COUNT(*) AS total_films,
        SUM(rental_duration) / COUNT(*) OVER () AS duration_sr
    FROM 
        temp1
    GROUP BY 
        name 
    ORDER BY 
        name
 """       
my_file_making(3, s)

#четвертый запрос - я не очень понял смысл задания, в базе нет свежих данных по продажам, поэтому просто использовал имеющуюся в базе вьюху...
df = spark.read.csv('tables_csv/sales_by_film_category.csv', header=True, sep=",")
df.createOrReplaceTempView("temp1")
s = """
    SELECT 
        *
    FROM 
        temp1
 """       
my_file_making(4, s)

#пятый запрос
df = spark.read.csv('tables_csv/sales_by_store.csv', header=True, sep=",")
df.createOrReplaceTempView("temp1")
s = """
    SELECT 
        *
    FROM 
        temp1
 """       
my_file_making(5, s)

#шестой запрос
df_category = spark.read.csv('tables_csv/category.csv', header=True, sep=",").drop("last_update")
df_film_category = spark.read.csv('tables_csv/film_category.csv', header=True, sep=",").drop("last_update")
df_res = df_category.join(df_film_category, ["category_id"])
df_film = spark.read.csv('tables_csv/film.csv', header=True, sep=",").drop("title", "description", "release_year", "last_update", "language_id", "original_language_id", "length", "rental_rate", "special_features",  "rating", "rental_duration")
df_res = df_res.join(df_film, ["film_id"])
df_res.createOrReplaceTempView("temp1")
s = """
    SELECT 
        name,
        COUNT(*) AS total_films,
        SUM(replacement_cost) / COUNT(*) OVER () AS replacement_sr
    FROM 
        temp1
    GROUP BY 
        name 
    ORDER BY 
        name
 """       
my_file_making(6, s)

#седьмой запрос
df_category = spark.read.csv('tables_csv/category.csv', header=True, sep=",").drop("last_update")
df_film_category = spark.read.csv('tables_csv/film_category.csv', header=True, sep=",").drop("last_update")
df_res = df_category.join(df_film_category, ["category_id"])
df_film = spark.read.csv('tables_csv/film.csv', header=True, sep=",").drop("replacement_cost", "title", "description", "release_year", "last_update", "language_id", "original_language_id", "length", "rental_rate", "special_features",  "rating", "rental_duration")
df_res = df_res.join(df_film, ["film_id"])
df_film_actor = spark.read.csv('tables_csv/film_actor.csv', header=True, sep=",").drop("last_update")
df_res = df_res.join(df_film_actor, ["film_id"])
df_actor = spark.read.csv('tables_csv/actor.csv', header=True, sep=",").drop("last_update")
df_res = df_res.join(df_actor, ["actor_id"])
df_res = df_res.select("name", concat_ws(" ", col("first_name"), col("last_name")).alias("full_name"))
df_res.createOrReplaceTempView("temp1")
s = """
    SELECT 
        COUNT(DISTINCT name) AS genre_value,
        full_name
    FROM 
        temp1
    GROUP BY
        full_name
    ORDER by
        full_name    
 """       
df1 = spark.sql(s)
df1.createOrReplaceTempView("temp1")
s = """
    SELECT 
        *,
        genre_value * 0 as y
    FROM 
        temp1
    WHERE
        genre_value > 15
 """       
my_file_making(7, s)

print('Файлы сформированы в папке results')

#визуализация запроса 2.1
df = pd.read_csv('results/2_1/res_2_1.csv')
sns.barplot(data = df, x = 'rating', y = 'percentage', palette = 'Set2')
plt.title('Доля фильмов разных категорий в нашем ассортименте')
plt.ylabel('Доля фильмов в %')
plt.xlabel('Рейтинг')
plt.show()

#визуализация запроса 2.2
df = pd.read_csv('results/2_2/res_2_2.csv')
sns.barplot(data = df.sort_values(by = 'count'), x = 'rating', y = 'count', palette = 'Set2')
plt.title('Категории фильмов, которые арендуются чаще всего')
plt.ylabel('Частота аренды фильмов')
plt.xlabel('Рейтинг')
plt.show()

#визуализация запроса 2.3
df = pd.read_csv('results/2_3/res_2_3.csv')
sns.barplot(data = df, x = 'name', y = 'duration_sr', palette = 'Set2')
plt.title('Средняя продолжительность проката для каждой категории фильмов')
plt.ylabel('Средняя продолжительность проката фильма, дни')
plt.xlabel('Категория')
plt.xticks(rotation = 45)
plt.show()

#визуализация запроса 2.4
df = pd.read_csv('results/2_4/res_2_4.csv')
sns.lineplot(data = df, x = 'category', y = 'total_sales')
plt.title('Продажи фильмов разных категорий')
plt.ylabel('Выручка, $')
plt.xlabel('Категория')
plt.xticks(rotation = 20)
plt.show()

#визуализация запроса 2.5
df = pd.read_csv('results/2_5/res_2_5.csv')
sns.barplot(data = df, x = 'store', y = 'total_sales', palette = 'Set2')
plt.title('Продажи в магазинах')
plt.ylabel('Продажи, $')
plt.xlabel('Магазины')
plt.show()

#визуализация запроса 2.6
df = pd.read_csv('results/2_6/res_2_6.csv')
sns.barplot(data = df, x = 'name', y = 'replacement_sr', palette = 'Set2')
plt.title('Средние затраты на замену фильмов')
plt.ylabel('Средние затраты, $')
plt.xlabel('Категория')
plt.xticks(rotation = 45)
plt.show()

#визуализация запроса 2.7
df = pd.read_csv('results/2_7/res_2_7.csv')
sns.barplot(data = df, x = 'y', y = 'full_name')
plt.title(' ')
plt.ylabel('Актеры, которые снялись в фильмах всех жанров')
plt.xlabel(' ')
plt.subplots_adjust(left = 0.6)
plt.show()