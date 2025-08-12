# Print first 10 natural numbers using while

print("1. First 10 Natural Numbers:")
i = 1
while i <= 10:
    print(i, end=" ")
    i += 1
print("\n")

# Take Input from user, and print even numbers till that input number

n = int(input("Enter a number for even numbers: "))
print("2. Even Numbers till", n, ":")
for i in range(2, n+1, 2):
    print(i, end=" ")
print("\n")

# Take Input from user, and print odd numbers till that input number

n = int(input("Enter a number for odd numbers: "))
print("3. Odd Numbers till", n, ":")
for i in range(1, n+1, 2):
    print(i, end=" ")
print("\n")

# Take Input from user, and print prime numbers till that input number

n = int(input("Enter a number for prime numbers: "))
print("4. Prime Numbers till", n, ":")
for num in range(2, n+1):
    is_prime = True
    for j in range(2, int(num**0.5) + 1):
        if num % j == 0:
            is_prime = False
            break
    if is_prime:
        print(num, end=" ")
print("\n")

# Print multiplication table of a given number

num = int(input("Enter a number for multiplication table: "))
print(f"5. Multiplication Table of {num}:")
for i in range(1, 11):
    print(f"{num} x {i} = {num * i}")


