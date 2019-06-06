def string_alternative(my_string):
    
    return my_string[::2]


my_string = input("Please enter a string")
final = string_alternative(my_string)
print(final)