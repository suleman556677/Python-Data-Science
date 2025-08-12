# Reverse the tuple

my_tuple = (10, 20, 30, 40, 50)
reversed_tuple = my_tuple[::-1]
print("1. Reversed Tuple:", reversed_tuple)

# Access value 20 from the tuple

nested_tuple = ("a", "b", (10, 20, 30), "c")
value_20 = nested_tuple[2][1]
print("2. Accessed Value 20:", value_20)

# Swap two tuples

tuple1 = (1, 2, 3)
tuple2 = (4, 5, 6)
tuple1, tuple2 = tuple2, tuple1
print("3. After Swapping:")
print("   Tuple1:", tuple1)
print("   Tuple2:", tuple2)
