import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Load CSV (full path to avoid FileNotFoundError)
df = pd.read_csv("Medical-Industry-pk\Medical-Industry-Pakistan.csv",index_col="Years")

# Convert number strings with commas to float
for col in df.columns:
    df[col] = df[col].replace({',': ''}, regex=True).astype(float)

# Display basic info
print("=== DataFrame ===")
print(df)
print("=== .info() ===")
print(df.info())
print("=== .dtypes ===")
print(df.dtypes)
print("=== .describe() ===")
print(df.describe())
print("=== .shape ===")
print(df.shape)

# Features & target
X = df[["Female Doctors"]].values
y = df["Female Dentists"].values

# Train-test split (70% / 30%)
SEED = 42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.70, random_state=SEED
)

# Train linear regression model
reg = LinearRegression()
reg.fit(X_train, y_train)

# Model parameters
print("Intercept (b0):", reg.intercept_)
print("Slope (b1):", reg.coef_[0])

# Prediction function
def calc_female_dentists(slope, intercept, female_doctors):
    return slope * female_doctors + intercept

# Predict for some training values
sample_values = [X_train[0][0], X_train[1][0], X_train[2][0]]
for val in sample_values:
    pred = calc_female_dentists(reg.coef_[0], reg.intercept_, val)
    print(f"Predicted Female Dentists for {val} Female Doctors: {pred:.2f}")

# Predict on test set
y_pred = reg.predict(X_test)
print("=== Predictions ===")
df_preds = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
print(df_preds)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("=== Evaluation Metrics ===")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Visualization
plt.scatter(X_test, y_test, color="blue", label="Actual")
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Predicted Line")
plt.xlabel("Female Doctors")
plt.ylabel("Female Dentists")
plt.title("Linear Regression: Female Doctors vs Female Dentists")
plt.legend()
plt.show()
print("END")