import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load CSV into DataFrame with "property_id" as index
df = pd.read_csv("Zameencom-Property-Data\zameencom-property-data-By-Kaggle-Short.csv",   delimiter=";" ,index_col="property_id" )

# Print DataFrame
print("=== DataFrame ===")
print(df)

# Call DataFrame methods/properties
print("=== .info() ===")
print(df.info())

print("=== .dtypes ===")
print(df.dtypes)

print("=== .describe() ===")
print(df.describe())

print("=== .shape ===")
print(df.shape)

# Convert to array format for sklearn
X = df[["bedrooms"]].values  #
y = df["price"].values       

# Split data (75% train, 25% test)
SEED = 32
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.75, random_state=SEED
)

# Train Linear Regression Model
reg = LinearRegression()
reg.fit(X_train, y_train)

# Intercept
print("\nIntercept (b0):", reg.intercept_)

# Slope
print("Slope (b1):", reg.coef_[0])

# Function to calculate price
def calc_price(slope, intercept, bedrooms):
    return slope * bedrooms + intercept

# Example: Predict for 3 bedrooms
bedroom_values = [3, 5, 2]  # you can pick from your dataset
for b in bedroom_values:
    price = calc_price(reg.coef_[0], reg.intercept_, b)
    print(f"Predicted price for {b} bedrooms: {price:.2f}")

# Predict for testing data
y_pred = reg.predict(X_test)
print("\n=== Predictions ===")
df_preds = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
print(df_preds)

# Metric analysis
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("=== Evaluation Metrics ===")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Optional: Visualization
plt.scatter(X_test, y_test, color="blue", label="Actual")
plt.plot(X_test, y_pred, color="red", label="Predicted")
plt.xlabel("Bedrooms")
plt.ylabel("Price")
plt.title("Linear Regression: Bedrooms vs Price")
plt.legend()
plt.show()
print('END')

