from itertools import permutations

def TMNT(s:str, n:int):
    return sorted(list(permutations(s, n)))

print(TMNT('abc', 2))