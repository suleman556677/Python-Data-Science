# Converts a temperature from Celsius to Fahrenheit.
# Formula: Fahrenheit = (Celsius * 9/5) + 32

celsius = float(input("Enter temperature in Celsius: "))
fahrenheit = (celsius * 9 / 5) + 32
print(f"Temperature in Fahrenheit: {fahrenheit:.2f}")

# Calculate Area of a Rectangle

length = float(input("Enter length of rectangle: "))
width = float(input("Enter width of rectangle: "))
area = length * width
print(f"Area of rectangle: {area}")

# Calculate Compound Interest
# Formula: CI = P * (1 + R/100)**T - P

P = float(input("Enter principal amount: "))
R = float(input("Enter annual interest rate (%): "))
T = float(input("Enter time in years: "))
CI = P * (1 + R / 100) ** T - P
print(f"Compound Interest: {CI:.2f}")

# Perimeter of a Rectangle

length = float(input("Enter length: "))
width = float(input("Enter width: "))
perimeter = 2 * (length + width)
print(f"Perimeter of rectangle: {perimeter}")


# Average of Three Numbers

num1 = float(input("Enter first number: "))
num2 = float(input("Enter second number: "))
num3 = float(input("Enter third number: "))
average = (num1 + num2 + num3) / 3
print(f"Average: {average}")


# Square and Cube of a Number

num = float(input("Enter a number: "))
print(f"Square: {num ** 2}")
print(f"Cube: {num ** 3}")


# Distribute Items Equally

n = int(input("Enter total candies: "))
k = int(input("Enter total students: "))
each = n // k
left = n % k
print(f"Each student gets: {each} candies")
print(f"Candies left: {left}")


# Calculate Profit or Loss

cp = float(input("Enter cost price: "))
sp = float(input("Enter selling price: "))

if sp > cp:
    print(f"Profit: {sp - cp}")
elif cp > sp:
    print(f"Loss: {cp - sp}")
else:
    print("No Profit No Loss")


# Total Marks and Percentage

marks = []
for i in range(1, 6):
    m = float(input(f"Enter marks of subject {i}: "))
    marks.append(m)

total = sum(marks)
percentage = (total / (5 * 100)) * 100
average = total / 5

print(f"Total Marks: {total}")
print(f"Percentage: {percentage:.2f}%")
print(f"Average Marks: {average}")


# Salary Calculator

basic = float(input("Enter basic salary: "))
HRA = basic * 0.20
DA = basic * 0.15
total_salary = basic + HRA + DA
print(f"HRA: {HRA}")
print(f"DA: {DA}")
print(f"Total Salary: {total_salary}")


# Age in Months and Days

years = int(input("Enter your age in years: "))
months = years * 12
days = years * 365  # Approximate
print(f"Age in months: {months}")
print(f"Age in days: {days}")


# Currency Converter (USD to PKR)

usd = float(input("Enter amount in USD: "))
rate = 280  # Fixed exchange rate
pkr = usd * rate
print(f"{usd} USD = {pkr} PKR")


# Sum of First N Natural Numbers
# Formula: sum = n * (n + 1) / 2

n = int(input("Enter a number: "))
total = n * (n + 1) // 2
print(f"Sum of first {n} natural numbers: {total}")


# Percentage of Correct Answers

total_q = int(input("Enter total questions: "))
correct = int(input("Enter correct answers: "))
percentage = (correct / total_q) * 100
print(f"Percentage score: {percentage:.2f}%")


# Speed, Distance, and Time

distance = float(input("Enter distance (km): "))
time = float(input("Enter time (hours): "))
speed = distance / time
print(f"Speed: {speed} km/h")


# Calculate Body Mass Index (BMI)

weight = float(input("Enter weight (kg): "))
height = float(input("Enter height (m): "))
BMI = weight / (height ** 2)
print(f"Your BMI is: {BMI:.2f}")


# Convert Minutes to Hours and Minutes

minutes = int(input("Enter total minutes: "))
hours = minutes // 60
remaining = minutes % 60
print(f"{minutes} minutes = {hours} hours {remaining} minutes")

