import numpy as np
import pandas as pd
from scipy import stats

# -----------------------------
# 1. Load CSV into NumPy Arrays
# -----------------------------
brokered_by, price, acre_lot, city, house_size = np.genfromtxt("RealEstate-USA.csv", delimiter=',', usecols=[0 , 2 , 5 , 7, 10] , dtype=None, unpack=True , skip_header=1,)

print(brokered_by)
print(price)
print(acre_lot)
print(city)
print(house_size)

print("=== Data Loaded ===")
print("Price array:", price[:5])
print("House size array:", house_size[:5])
print()

# --------------------------------------------
# 2. Descriptive Statistics for 'price'
# --------------------------------------------
print("=== Descriptive Statistics: PRICE ===")
print("Mean:", np.mean(price))
print("Median:", np.median(price))
print("Mode:", stats.mode(price, keepdims=True))
print("Standard Deviation:", np.std(price))
print("Variance:", np.var(price))
print("Max:", np.max(price))
print("Min:", np.min(price))
print()

# --------------------------------------------
# 3. Descriptive Statistics for 'house_size'
# --------------------------------------------
print("=== Descriptive Statistics: HOUSE SIZE ===")
print("Mean:", np.mean(house_size))
print("Median:", np.median(house_size))
print("Mode:", stats.mode(house_size, keepdims=True))
print("Standard Deviation:", np.std(house_size))
print("Variance:", np.var(house_size))
print("Max:", np.max(house_size))
print("Min:", np.min(house_size))
print()

# ---------------------------------------------------------
# 4. Arithmetic Operations on price & house_size arrays
# ---------------------------------------------------------
print("=== Arithmetic Operations ===")
print("Addition (+):", price + house_size)
print("Addition (np.add):", np.add(price, house_size))
print("Subtraction (-):", price - house_size)
print("Subtraction (np.subtract):", np.subtract(price, house_size))
print("Multiplication (*):", price * house_size)
print("Multiplication (np.multiply):", np.multiply(price, house_size))
print()

# ----------------------------------------
# 5. Create 2D Array [price, house_size]
# ----------------------------------------
two_d_array = np.array([price, house_size])
print("=== 2D Array ===")
print(two_d_array)
print()

# ----------------------------------------
# 6. Create 3D Array [house_size, price, acre_lot]
# ----------------------------------------
three_d_array = np.array([house_size, price, acre_lot])
print("=== 3D Array ===")
print(three_d_array)
print()

# ----------------------------------------
# 7. Iterate using np.nditer
# ----------------------------------------
print("=== Iteration with np.nditer (PRICE) ===")
for val in np.nditer(price):
    print(val)
print()

# ----------------------------------------
# 8. Iterate using np.ndenumerate
# ----------------------------------------
print("=== Iteration with np.ndenumerate (PRICE) ===")
for idx, val in np.ndenumerate(price):
    print(f"Index {idx}: Value {val}")
print()

# ----------------------------------------
# 9. 7 Common Properties of PRICE Array
# ----------------------------------------
print("=== Array Properties (PRICE) ===")
print("ndim:", price.ndim)
print("shape:", price.shape)
print("size:", price.size)
print("dtype:", price.dtype)
print("itemsize:", price.itemsize)
print("nbytes:", price.nbytes)
print("strides:", price.strides)
print()

# ---------------------------------------------------
# 10. Slice 2D Array (Rows 1:3, Columns 2:4)
# ---------------------------------------------------
print("=== Slice 1 (Rows 1 to 3, Cols 2 to 4) ===")
print(two_d_array[1:3, 2:4])
print()

# ---------------------------------------------------
# 11. Slice 2D Array (Rows 2:8, Cols 3:5)
# ---------------------------------------------------
print("=== Slice 2 (Rows 2 to 8, Cols 3 to 5) ===")
print(two_d_array[2:8, 3:5])
print()

# ---------------------------------------------------
# 12. Geometric Operations on 2D Array
# ---------------------------------------------------
print("=== Geometric Operations (2D Array) ===")
print("Sine:\n", np.sin(two_d_array))
print("Cosine:\n", np.cos(two_d_array))
print("Tangent:\n", np.tan(two_d_array))
print("Arcsine:\n", np.arcsin(np.clip(two_d_array, -1, 1)))
print("Arccosine:\n", np.arccos(np.clip(two_d_array, -1, 1)))
print("Arctangent:\n", np.arctan(two_d_array))
