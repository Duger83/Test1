import shutil
import os
from pathlib import Path

class safe_write:
    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        if Path(self.filename).exists(): 
            shutil.copy(self.filename, self.filename+'_pro')
        self.file = open(self.filename, mode='w')
        return self.file

    def __exit__(self, exc_type, exc_value, traceback):
        if self.file:
            self.file.close()
        if exc_value != None:
            my_exc_type = str(exc_type).replace(str(exc_type)[:8], "", 1)
            my_exc_type = my_exc_type[:-2]
            print('Во время записи в файл было возбуждено исключение ' + my_exc_type)
            if Path(self.filename+'_pro').exists(): 
                shutil.copy(self.filename+'_pro', self.filename)
            if Path(self.filename+'_pro').exists(): 
                os.remove(self.filename+'_pro')       
            return True
        else:
            if Path(self.filename+'_pro').exists():
                os.remove(self.filename+'_pro')                                
            
with safe_write('undertale.txt') as file:
    file.write('Я знаю, что ничего не знаю, но другие не знают и этого.\n')

with safe_write('undertale.txt') as file:
    print(
            'Если ты будешь любознательным, то будешь много знающим.',
            file=file,
            flush=True
        )
    raise ValueError

with open('undertale.txt') as file:  
    print(file.read())