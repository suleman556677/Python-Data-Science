# Check if a value exists in a dictionary

my_dict = {"a": 10, "b": 20, "c": 30}
value_to_check = 20
if value_to_check in my_dict.values():
    print(f"1. Value {value_to_check} exists in the dictionary")
else:
    print(f"1. Value {value_to_check} does NOT exist in the dictionary")

# Get the key of a minimum value

min_key = min(my_dict, key=my_dict.get)
print("2. Key with Minimum Value:", min_key)

# Delete a list of keys from a dictionary

keys_to_delete = ["a", "c"]
for key in keys_to_delete:
    my_dict.pop(key, None) 
print("3. Dictionary after deleting keys:", my_dict)
