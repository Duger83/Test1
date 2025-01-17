import types

def generate_complex_function(function_name, parameters, function_body):
    s = 'def ' + function_name + '('
    for i in parameters:
        s = s + i +', '
    s = s[:-2] + '): ' + function_body
    namespace = {}
    exec(s, namespace)
    return types.FunctionType(namespace[function_name].__code__, globals())    
        
function_name = 'complex_function'
parameters = ['x', 'y']
function_body = """
    if x > y:
        return x - y
    else:
        return y - x
"""
complex_func = generate_complex_function(function_name, parameters, function_body)
print(complex_func(10, 5))
print(complex_func(5, 10))