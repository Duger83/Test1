# из задания не ясно, как именно пользователь вводит в программу список
# поэтому запрашиваем у пользователя список c клавиатуры (получаем список строк)
input_list = [i for i in input().split()]

#пытаемся сделать в полученном списке строковые значения числами, 
# не сгенерировав при этом исключение...
my_list=[]
for i in range(len(input_list)):
    try:
        my_list.append(float(input_list[i]))
    except:
        my_list.append(input_list[i])
 
#отвечаем на вопросы задачи 1            
if any(i > 0 for i in my_list if isinstance(i, float)):
    print('В введённом списке есть положительное число')
else:
    print('В введенном списке нет положительных чисел')

if all(isinstance(i, (int,float)) for i in my_list):
    print('В введенном списке только числа')
else:
    print('В введённом списке есть не только числа')

print('Отсортированный список -', sorted(input_list))