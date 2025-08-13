import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load csv file into DataFrame
df = pd.read_csv("C:/Users/Dell/OneDrive/Documents/Multiple-Linear-Regression/50_Startups (1).csv") 
print("DataFrame:")
print(df)

# -------------------------
# Inspect Data
# -------------------------
print("Info:")
print(df.info())

print("Data Types:")
print(df.dtypes)

print("Describe:")
print(df.describe())

print("Shape:", df.shape)

# -------------------------
# Independent & Dependent Variables
# Profit = dependent (y)
# R&D Spend, Administration, Marketing Spend = independent (X)
# -------------------------
# Drop non-numeric column like 'State'
df_numeric = df.select_dtypes(include=[np.number])

X = df_numeric.iloc[:, :-1].values  # all columns except last
y = df_numeric.iloc[:, -1].values   # last column is Profit

print("\nIndependent Variables Array (X):\n", X[:5])
print("\nDependent Variable Array (y):\n", y[:5])

# -------------------------
# Best-fitting regression lines
# -------------------------
for col in df_numeric.columns[:-1]:
    sns.regplot(x=df_numeric[col], y=df_numeric["Profit"], ci=None, line_kws={"color": "red"})
    plt.title(f"Regression Line: {col} vs Profit")
    plt.show()

# -------------------------
# Correlation Heatmap
# -------------------------
plt.figure(figsize=(8, 6))
sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# -------------------------
# Split Data (90% train, 10% test)
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# -------------------------
# Train Linear Regression Model
# -------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------
# Intercept
# -------------------------
print("Intercept (b0):", model.intercept_)

# -------------------------
# Slopes (Coefficients)
# -------------------------
print("Slopes (b1, b2, b3):", model.coef_)

# -------------------------
# Predictions
# -------------------------
y_pred = model.predict(X_test)
print("Predicted Profit:", y_pred)

# -------------------------
# Metric Analysis
# -------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)

