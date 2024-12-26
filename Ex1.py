import functools
user_role = ['admin']
def role_required(role: str):
    global user_role
    def check_permission(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    if role in user_role:
                        return func(*args, **kwargs)
                    else:
                        raise PermissionError
                except PermissionError as er:
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