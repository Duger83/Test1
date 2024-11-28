import pandas as pd

class Movie():

    def __init__(self, tit, rasp):         
        self.tit = tit
        self.rasp = iter(rasp)
    
    def schedule(self):
        flag = True
        while flag:
            try:
                a = next(self.rasp)
                prokat = pd.date_range(a[0].replace(',','-'), a[1].replace(',','-')).to_list()  
                yield prokat
            except:
                flag = False 
            
#задаём периоды дат показа фильма, только проще, чем в задании
my_list = [('2024,11,1', '2024,11,3'), 
           ('2024,12,15', '2024,12,18'),
           ('2024,12,29', '2024,12,31')]

film = Movie('ML-engineer', my_list)

flag1 = True
while flag1:
    try:
        pokaz = next(film.schedule())
        for val in pokaz:
            print(val.date())
    except:
        flag1 = False