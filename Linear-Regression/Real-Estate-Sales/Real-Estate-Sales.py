import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load the CSV with 'Serial Number' as index
df = pd.read_csv("Real-Estate-Sales\Real_Estate_Sales_2001-2022_GL-Short (1).csv",index_col="Serial Number")

print(df)

# Inspect DataFrame
print(df.info())
print(df.dtypes)
print(df.describe())
print(df.shape)

# Prepare features and labels
X = df[["Assessed Value"]].values  # shape (n_samples, 1)
y = df["Sale Amount"].values       # shape (n_samples,)

# Split into 90% train, 10% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42
)

# Train Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Intercept
intercept = model.intercept_
print("Intercept:", intercept)

# Slope (coefficient)
slope = model.coef_[0]
print("Slope:", slope)

# Function to predict Sale Amount
def predict_sale_amount(assessed_value):
    return intercept + slope * assessed_value

# Test function with three sample values (picked from dataset)
sample_values = df["Assessed Value"].iloc[:3].tolist()
for val in sample_values:
    print(f"Estimated Sale Amount for Assessed Value {val}: {predict_sale_amount(val)}")

# Predict using testing data
y_pred = model.predict(X_test)

# Metrics: MAE, MSE, RMSE
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)

# Visualize predictions vs actual values
# === Visualization with Seaborn ===
plt.figure(figsize=(8, 6))
sns.regplot(x="Assessed Value", y="Sale Amount", data=df, line_kws={"color": "red"}, scatter_kws={"alpha": 0.5})
plt.title("Assessed Value vs Sale Amount (Linear Regression)", fontsize=14)
plt.xlabel("Assessed Value")
plt.ylabel("Sale Amount")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
print("END")
