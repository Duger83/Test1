expiration = 6

def cache(db: str): 
    global expiration
    def my_func(func):
        global expiration
        def wrapper(*args, **kwargs):
            global expiration
            expiration -= 1
            if expiration < 5:
                print("Info about: " + db + " cached in postgresql, expire=" + str(expiration))
            if expiration == 5:
                print("Info about: " + db + " from postgresql, now cached with expire=" + str(expiration))
            if expiration == 0:
                expiration = 6
        return wrapper
    return my_func

print()

for i in range(15):
    @cache('bike_store')
    def get_info(thing: str):
        print('Info about: ' + thing)
    get_info()    