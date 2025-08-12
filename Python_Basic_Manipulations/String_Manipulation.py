import string

# Input string

str1 = "Python-is-awesome!!!"

print("Original String:", str1)

# Create new string from first, middle, last char

first_char = str1[0]
middle_char = str1[len(str1) // 2]
last_char = str1[-1]
new_str = first_char + middle_char + last_char
print("1. First, Middle & Last Char String:", new_str)

# Count occurrences of all characters

char_count = {}
for char in str1:
    char_count[char] = char_count.get(char, 0) + 1
print("2. Character Occurrences:", char_count)

# Reverse the string

reversed_str = str1[::-1]
print("3. Reversed String:", reversed_str)

# Split string on hyphens

parts = str1.split('-')
print("4. Split on Hyphens:", parts)

# Remove punctuation/special symbols

clean_str = str1.translate(str.maketrans('', '', string.punctuation))
print("5. String without Punctuation:", clean_str)
