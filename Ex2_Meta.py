def create_class_with_methods(class_name, class_attributes, class_methods):
    DynClassAttr = type(class_name + '_atrr', (), class_attributes)
    DynClassMethod = type(class_name, (DynClassAttr, ), class_methods)
    return DynClassMethod
  
attributes = { 'species': 'Human', 'age': 25 }
methods = { 'greet': lambda self: f"Hello, I am a {self.species} and I am {self.age} years old." }

DynamicClass = create_class_with_methods('DynamicClass', attributes, methods)
instance = DynamicClass()
print(instance.greet())