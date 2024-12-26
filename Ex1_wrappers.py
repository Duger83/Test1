user_role = ['admin']

def role_required(role: str): 
    global user_role
    def check_permission(func):
        def wrapper(*args, **kwargs):
            if role in user_role:
                return func(*args, **kwargs)
            else:
                print('Permission denied')
        return wrapper
    return check_permission

print()

print('Проверка доступности функции secret_source() для пользователя admin:')
@role_required('admin')
def secret_source():
    print('Permission accepted')
secret_source()    

print()

print('Проверка доступности функции secret_source() для пользователя manager:')
@role_required('manager')
def secret_source():
    print('Permission accepted')
secret_source()

print()