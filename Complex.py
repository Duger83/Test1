import math

# для задания числа в алгебраической форме      x = Compl(Re, Im, 'alg')
# для задания числа в тригонометрической форме  х = Compl(R, φ, 'pol')

class Compl(object):
    def __init__(self, re, im, mode): 
        self.mode = mode
        if mode == "alg":
            self.re = re
            self.im = im
        if mode == "pol":
            self.re = re * math.cos(im)
            self.im = re * math.sin(im)

# преобразование числа в тригонометрическую форму   
    def polar(self):
        a = self.re
        b = self.im
        self.re = math.sqrt(a**2 + b**2)
        if a == 0:
            if b == 0:
                self.im = 0
            else:
                self.im = math.pi / 2
        else:
            self.im = math.atan(b / a)
        self.mode = "pol"
        return self

# все вычисления делаем в алгебраической форме
    def __add__(self, no):
        return Compl(self.re + no.re, self.im + no.im, "alg")

    def __sub__(self, no):
        return Compl(self.re - no.re, self.im - no.im, "alg")

    def __mul__(self, no):
        return Compl(self.re * no.re - self.im * no.im, self.re * no.im + self.im * no.re, "alg")

    def __truediv__(self, no):
        if (no.re**2 + no.im**2) == 0:
            return Compl(0 ,0 , "alg")
        else:
            return Compl((self.re * no.re + self.im * no.im)/(no.re**2 + no.im**2), (self.im * no.re - self.re * no.im)/(no.re**2 + no.im**2), "alg")

    def __str__(self):
        if self.mode=="alg":
            if self.im >= 0:
                result = "(%.2f+%.2fi)" % (self.re, self.im)
            else:
                result = "(%.2f-%.2fi)" % (self.re, abs(self.im))
        if self.mode=="pol":
            result = "%.2f*(cos(%.2f)+isin(%.2f))" % (self.re, self.im, self.im)
        return result

x = Compl(1, 0, 'alg')
y = Compl(1, 0, 'pol')
c = y
c.mode='alg'

print("в алгеб. форме Х =", x)
print("в триг.  форме X =", Compl.polar(x))
print("в алгеб. форме Y =", c)
print("в триг.  форме Y =", Compl.polar(y))
print("в алгеб. форме Х + Y =", x + y)
print("в триг.  форме X + Y =", Compl.polar(x + y))
print("в алгеб. форме Х - Y =", x - y)
print("в триг.  форме X - Y =", Compl.polar(x - y))
print("в алгеб. форме Х * Y =", x * y)
print("в триг.  форме X * Y =", Compl.polar(x * y))
print("в алгеб. форме Х / Y =", x / y)
print("в триг.  форме X / Y =", Compl.polar(x / y))