import psycopg2
res = open('result_queries_ex1.txt', 'w')
connection = psycopg2.connect(dbname="BikeStore", user="postgres", password="04011983", host="127.0.0.1", port="5432")

cursor = connection.cursor()

def my_print():
    global res
    global cursor
    my_table = cursor.fetchall()
    s = "["
    for row in cursor.description:
        s = s + row[0] + ", "
    s = s[:-2] + "]"
    print(s, file = res)
    for row in my_table:
        print(row, file = res)
    print(" ", file = res)
    print(" ", file = res)

#первый запрос
print("Получение названий всех продуктов и соответствующих им торговых марок:", file = res)
my_query = """SELECT products.product_name, brands.brand_name
                FROM products 
                JOIN brands ON products.brand_id = brands.brand_id"""
cursor.execute(my_query)
connection.commit()
my_print()

#второй запрос
print("Нахождение всех активных сотрудников и наименований магазинов, в которых они работают:", file = res)
my_query = """SELECT staffs.last_name, staffs.first_name, stores.store_name
                FROM staffs 
                JOIN stores ON ((staffs.active = 1) and (staffs.store_id = stores.store_id))"""
cursor.execute(my_query)
connection.commit()
my_print()

#третий запрос
print("Перечисление всех покупателей выбранного магазина с указанием их полных имен, адреса электронной почты и номера телефона:", file = res)
my_query = """SELECT distinct on (c.last_name) c.last_name, c.first_name, c.email, c.phone, s.store_name
                FROM customers c 
                INNER JOIN orders o ON o.store_id = 1 and c.customer_id = o.customer_id
                INNER JOIN stores s ON s.store_id = o.store_id"""
cursor.execute(my_query)
connection.commit()
my_print()

#четвертый запрос
print("Подсчет количества продуктов в каждой категории:", file = res)
my_query = """SELECT c.category_name, count(*) AS count_products
                FROM products p
                JOIN categories c ON c.category_id = p.category_id
                GROUP BY c.category_name"""
cursor.execute(my_query)
connection.commit()
my_print()

#пятый запрос
print("Общее количество заказов для каждого клиента:", file = res)
my_query = """SELECT c.last_name, c.first_name, count(*) AS count_orders
                FROM orders o
                JOIN customers c ON c.customer_id = o.customer_id
                GROUP BY c.customer_id"""
cursor.execute(my_query)
connection.commit()
my_print()

#шестой запрос
print("Информация о полном имени и общем количестве заказов клиентов, которые больше одного раза сделали заказ:", file = res)
#немного изменил формулировку, иначе вывод получался такой же, как в запросе №5, а это не интересно...
my_query = """SELECT c.last_name, c.first_name, count(*) AS count_orders
                FROM orders o
                JOIN customers c ON c.customer_id = o.customer_id
                GROUP BY c.customer_id
                HAVING count(*) > 1"""
cursor.execute(my_query)
connection.commit()
my_print()

connection.close()
res.close()
print("Создан файл результатов запросов - result_queries_ex1.txt")