import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# =======================
# Load CSV
# =======================
df = pd.read_csv(r"C:\Users\Dell\OneDrive\Documents\Stock-Prices\Stock-Prices.csv.csv")
print("Loaded data. Shape:", df.shape)

# Drop NaN values
df.dropna(how="all", inplace=True)
print("After removing empty rows:", df.shape)

print("CSV Columns:", df.columns.tolist())

# =======================
# Detect Target Column
# =======================
preferred_targets = ['close', 'adj close', 'price']
target_col = None
for col in df.columns:
    if col.lower() in preferred_targets:
        target_col = col
        break

if target_col is None:
    print("No valid target column found. Available columns:", df.columns.tolist())
    exit()

print("Target column:", target_col)

# =======================
# Drop Irrelevant Columns
# =======================
irrelevant_cols = ['date', 'open', 'high', 'low', 'ticker', 'name']
df.drop(columns=[col for col in df.columns if col.lower() in irrelevant_cols], inplace=True, errors='ignore')

# Convert target to numeric
df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
df.dropna(subset=[target_col], inplace=True)

# =======================
# Select Features
# =======================
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
X_cols = [col for col in numeric_cols if col != target_col]
X = df[X_cols]
y = df[target_col]

print("Final X shape:", X.shape)
print("Final y shape:", y.shape)

if X.empty:
    print("No data available for training after preprocessing.")
    print("Numeric columns detected:", numeric_cols)
    exit()

# =======================
# Scale Data
# =======================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =======================
# Train/Test Split
# =======================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# =======================
# Linear Regression Model
# =======================
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("Linear Regression R²:", r2_score(y_test, y_pred_lr))
print("Linear Regression MSE:", mean_squared_error(y_test, y_pred_lr))

# =======================
# Deep Learning Model
# =======================
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

y_pred_dl = model.predict(X_test).flatten()
print("Deep Learning MSE:", mean_squared_error(y_test, y_pred_dl))

# =======================
# Plot Predictions
# =======================
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lr, label="Linear Regression", alpha=0.6)
plt.scatter(y_test, y_pred_dl, label="Deep Learning", alpha=0.6, color="red")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Model Prediction Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =======================
# Correlation Matrix
# =======================
plt.figure(figsize=(10, 8))
sns.heatmap(df[X_cols + [target_col]].corr(), annot=True, cmap='coolwarm')
plt.title(
    "Correlation Matrix of Features and Target")
plt.tight_layout()
plt.show()

