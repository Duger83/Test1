class CiclicIterator():
    def __init__(self, iter_obj):         
        self.iter_obj = iter(iter_obj)
        self.prot = iter_obj
    
    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iter_obj)
        except StopIteration:
            self.iter_obj = iter(self.prot)
            return next(self.iter_obj)
            
my_list = [1, 2, 3]
my_tuple = (1, 2, 3)

a = CiclicIterator(my_list)  
b = CiclicIterator(my_tuple)

for i in range (10):
    print(next(a))

print()

for i in range(10):
    print(next(b)) 