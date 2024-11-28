import random
from string import ascii_lowercase, ascii_uppercase

chars = ascii_lowercase + ascii_uppercase + "0123456789!?@#$*"

def Passwords(base_str):
    while True:
        yield ("".join([random.choice(base_str) for i in range(12)]))

My_Pass = Passwords(chars)

for i in range(5):
    print(next(My_Pass))