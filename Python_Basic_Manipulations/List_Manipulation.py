# Reverse a list

my_list = [1, 2, 3, 4, 5]
print("Original List:", my_list)
my_list.reverse()
print("1. Reversed List:", my_list)

# Turn every item into its square

squared_list = [x ** 2 for x in my_list]
print("2. Squared List:", squared_list)

# Remove empty strings from the list of strings

string_list = ["apple", "", "banana", "", "cherry"]
filtered_list = [s for s in string_list if s != ""]
print("3. Without Empty Strings:", filtered_list)

# Add new item to list after a specified item

list_with_insert = ["apple", "banana", "cherry"]
item_to_insert_after = "banana"
new_item = "mango"
if item_to_insert_after in list_with_insert:
    index = list_with_insert.index(item_to_insert_after)
    list_with_insert.insert(index + 1, new_item)
print("4. After Adding Item:", list_with_insert)

# Replace list’s item with new value if found

replace_list = [10, 20, 30, 40]
old_value = 30
new_value = 99
if old_value in replace_list:
    index = replace_list.index(old_value)
    replace_list[index] = new_value
print("5. After Replacement:", replace_list)
