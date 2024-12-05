def my_len_str(s:str):
    return False if len(s)<3 else True

Cities_input = input('Введите список городов:\n').split(',')
Cities_res = [i for i in Cities_input if my_len_str(i)]
print('Список городов с длинными названиями: ',Cities_res)