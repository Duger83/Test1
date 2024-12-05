from functools import partial

def _sort_users_by_age(users_list:list, order):
    if order == 'up':
        res = sorted(users_list, key=lambda x: x['Age'])
    if order == 'down':
        res =  sorted(users_list, key=lambda x: x['Age'], reverse=True)       
    return res

up_sort_users_by_age = partial(_sort_users_by_age, order = 'up')
down_sort_users_by_age = partial(_sort_users_by_age, order = 'down')

user1 = {'Last_Name': 'Freud',
         'First_Name': 'Sigmund',
         'Age': 83}

user2 = {'Last_Name': 'Jung',
         'First_Name': 'Carl',
         'Age': 85}

user3 = {'Last_Name': 'Adler',
         'First_Name': 'Alfred',
         'Age': 67}

my_users = [user1, user2, user3]

print(up_sort_users_by_age(my_users))
print()
print(down_sort_users_by_age(my_users))