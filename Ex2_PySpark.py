from pyspark.sql import SparkSession 
from pyspark.sql.functions import to_date, date_trunc
spark = SparkSession.builder.master("local[*]").getOrCreate()
spark.conf.set("spark.sql.repl.eagerEval.enabled", True) 
print()

#первый запрос
print("Общий объем продаж по каждому продукту:")
df_orders = spark.read.csv('order_items.csv', header=True, sep=",").drop("order_id", "item_id")
df_products = spark.read.csv('products.csv', header=True, sep=",").drop("list_price", "brand_id", "category_id", "model_year")
df_res = df_orders.join(df_products, ["product_id"])
df_res.createOrReplaceTempView("temp1")
spark.sql("select product_name, round(sum(quantity * list_price * (1 - discount)), 2) as Total from temp1 group by product_id, product_name").show()

#второй запрос
print("Количество заказов с заданным статусом:")
df_orders = spark.read.csv('orders.csv', header=True, sep=",").drop("order_id", "order_date", "shipped_date", "required_date", "store_id", "staff_id")
df_orders.createOrReplaceTempView("temp1")
spark.sql("select order_status, count(*) as Total_count from temp1 group by order_status;").show()

#третий запрос
print("Расчет суммы продаж за месяц: (предполагаем, что заказ оформляется по предоплате)")
df_order_items = spark.read.csv('order_items.csv', header=True, sep=",").drop("item_id", "product_id")
df_orders = spark.read.csv('orders.csv', header=True, sep=",").drop("required_date", "shipped_date", "store_id", "staff_id", "order_status", "customer_id")
df_res = df_orders.join(df_order_items, ["order_id"])
df_res = df_res.withColumn('order_date', to_date(date_trunc('mon', to_date(df_res.order_date, 'yyyy-MM-dd')), 'yyyy-MM-dd'))
df_res.createOrReplaceTempView("temp1")
spark.sql("select order_date as Order_month, round(sum(quantity * list_price * (1 - discount)), 2) as Total from temp1 group by order_date").sort("order_date").show()

#четвертый запрос
print("5 самых \"дорогих\" клиентов")
df_order_items = spark.read.csv('order_items.csv', header=True, sep=",").drop("item_id", "product_id")
df_orders = spark.read.csv('orders.csv', header=True, sep=",").drop("required_date", "shipped_date", "store_id", "staff_id", "order_status", "order_date")
df_res = df_orders.join(df_order_items, ["order_id"])
df_customers = spark.read.csv('customers.csv', header=True, sep=",").drop("phone", "email", "city", "street", "state", "zip_code")
df_res = df_res.join(df_customers, ["customer_id"])
df_res.createOrReplaceTempView("temp1")
spark.sql("select last_name, first_name, round(sum(quantity * list_price * (1 - discount)), 2) as Total from temp1 group by customer_id, last_name, first_name").sort("Total", ascending=False).show(5)