import math

class Vector(object):
    
    def __init__(self, *a_list):
        if len(a_list)==0: self.val = (0)
        else:
            for i in range(len(a_list)):
                self.val=a_list[i]
          
    def __str__(self):
        return str(self.val)

    def __len__(self):
        return len(self.val)

    def __add__(a, b):
        if isinstance(b, Vector):
            if len(a)==len(b):
                added = tuple(i + j for i, j in zip(a.val, b.val))
            else:
                ValueError('Попытка сложить векторы разных размерностей. Складывать можно только векторы одинаковой размерности.')
        else:
            ValueError('Попытка сложить вектор и другой тип данных. Складывать можно только векторы.')
        return Vector(added)

    def __sub__(a, b):
        if isinstance(b, Vector):
            if len(a)==len(b):
                subed = tuple(i - j for i, j in zip(a.val, b.val))
            else:
                ValueError('Попытка вычесть векторы разных размерностей. Вычитать можно только векторы одинаковой размерности.')
        else:
            ValueError('Попытка вычесть из вектора другой тип данных. Вычитать можно только векторы.')
        return Vector(subed)  
    
    def dot_prod(a, b):
        if not isinstance(b, Vector):
            raise ValueError('Попытка скалярно перемножить вектор и другой тип данных. Скалярно перемножать можно только векторы.')
        if len(a) != len(b):
            raise ValueError('Попытка скалярно перемножить векторы разных размерностей. Скалярно перемножать можно только векторы одинаковой размерности.')
        return sum(i * j for i, j in zip(a.val, b.val))

    def __mul__(a, b):
        if isinstance(b, Vector):
            if len(a)==len(b):
                return Vector.dot_prod(a, b)
            else:
                raise ValueError('Попытка скалярно перемножить векторы разных размерностей. Скалярно перемножать можно только векторы одинаковой размерности.')
        elif isinstance(b, (int, float)):
            return Vector(tuple(i * b for i in a.val))
        else:
            raise ValueError('Попытка перемножить величины несовместимых типов. Перемножать можно только векторы и числа.')

    def norm(self):
        if not isinstance(self, Vector):
            raise ValueError('Попытка вычислить длину невекторной величины.')
        else:
            return math.sqrt(sum(i*i for i in self.val))

    def cosphi(a, b):
        if not isinstance(b, Vector):
            raise ValueError('Попытка вычислить угол между невекторными величинами.')
        else:
            if len(a)==len(b):
                return(Vector.dot_prod(a,b)/(Vector.norm(a)*Vector.norm(b)))
            else:
                ValueError('Попытка вычислить угол между векторами разных размерностей. Угол можно вычислить только между векторами одинаковой размерности.')

    
            
a = (9, 8, 7, 6, 5, 4, 3, 2, 1)
b = (1, 2, 3, 4, 5, 6, 7, 8, 9)
c = Vector(a)
d = Vector(b)

print('Вектор С =', c)
print('Вектор D =', d)
print('Вектор C + D =', c + d)
print('Вектор С - D =', c - d)
print('Скалярное произведение С * D =', c * d)
print('Умножение вектора С на число 2 =', c * 2)
print('|С| =', Vector.norm(c))
print('COS угла между векторами С и D =', Vector.cosphi(c, d), 'радиан')
print('При выполнении операций класс обрабатывает ошибки типов и размерностей.')