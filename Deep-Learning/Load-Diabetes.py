import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
diabetes = datasets.load_diabetes()
X = diabetes.data  # Features (10 variables)
y = diabetes.target  # Target (disease progression)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))


X_bmi = diabetes.data[:, np.newaxis, 2]


X_train_bmi, X_test_bmi, y_train_bmi, y_test_bmi = train_test_split(X_bmi, y, test_size=0.2, random_state=42)


model_bmi = LinearRegression()
model_bmi.fit(X_train_bmi, y_train_bmi)
y_pred_bmi = model_bmi.predict(X_test_bmi)


plt.scatter(X_test_bmi, y_test_bmi, color='black')
plt.plot(X_test_bmi, y_pred_bmi, color='blue', linewidth=2)
plt.xlabel("BMI")
plt.ylabel("Disease Progression")
plt.title("Linear Regression on BMI Feature")
plt.show()