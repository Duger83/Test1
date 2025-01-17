class AttrLoggingMeta(type):
    def __new__(cls, name, bases, dct):
        for key, value in dct.items():
            if not key.startswith('__'):
                dct[key] = cls.wrap_property(key, value)
        return super().__new__(cls, name, bases, dct)
    
    @staticmethod
    def log_access(name, value):
        print(f"Calling method {name}")
    
    @staticmethod
    def log_read(name, value, instance):
        print(f"Reading attribute {name}")
    
    @staticmethod
    def log_write(name, value, instance):
        print(f"Writing attribute {name} with value {value}")
    
    @classmethod
    def wrap_property(cls, name, value):
        if isinstance(value, property):
            prop = value
            
            def getter(instance):
                cls.log_read(name, prop.fget(instance), instance)
                return prop.fget(instance)
            
            def setter(instance, new_value):
                cls.log_write(name, new_value, instance)
                prop.fset(instance, new_value)
                
            return property(getter, setter)
        else:
            def wrap_method(method):
                def wrapper(*args, **kwargs):
                    cls.log_access(method, None)
                return wrapper
            return wrap_method(name)
        
class LoggedClass(metaclass=AttrLoggingMeta):
    def __init__(self, custom_method):
        self._custom_method = custom_method
    
    @property
    def custom_method(self):
        return self._custom_method
    
    @custom_method.setter
    def custom_method(self, new_value):
        self._custom_method = new_value 
        
    def other_custom_method(self):
        pass
    
instance = LoggedClass(42)
print(instance.custom_method)
instance.custom_method = 78
instance.other_custom_method()