from itertools import combinations

def Splinter(s:str, k:int):
    my_list = []
    for i in range(1, k+1):
        my_list.append(list(combinations(s, i)))
    return my_list
    
print(Splinter('abcd', 2))