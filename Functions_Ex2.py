def get_list(s:str):
    try:
        return [int(a) for a in s.split(' ')]
    except:
        return []

def sort_func(f, my_list):
    return sorted(f(my_list))
    
print(sort_func(get_list, input('Введите в строку целые числа через пробел:\n')))