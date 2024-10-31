try:
    a=int(input())
    b=int(input())
    c=int(input())
    numbers=[a,b,c]
    print()
    print(max(numbers))
    numbers.pop(numbers.index(max(numbers)))
    print(min(numbers))
    numbers.pop(numbers.index(min(numbers)))
    print(min(numbers))
except:
    print("Программа работает только с целыми числами. Введено НЕ целое число!")
    