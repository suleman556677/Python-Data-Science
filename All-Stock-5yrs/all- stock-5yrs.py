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

df = pd.read_csv("all_stock_5yrs.csv")
print("Loaded data. Shape:", df.shape)

df.dropna(inplace=True)
print(df.shape)

print("Columns:", df.columns.tolist())

preferred_targets = ['adj close', 'Adj Close', 'Adj_Close', 'close', 'Close']
target_col = None
for col in df.columns:
    if col.lower() in [t.lower() for t in preferred_targets]:
        target_col = col
        break

if target_col is None:
    print("No valid target column like 'Close' or 'Close' found. Exiting program.")
    exit()

print("Target column:", target_col)

irrelevant_cols = ['date', 'open', 'high', 'low', 'ticker']
for col in irrelevant_cols:
    if col in df.columns:
        df.drop(columns=col, inplace=True)

df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
df.dropna(subset=[target_col], inplace=True)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
X_cols = [col for col in numeric_cols if col != target_col]

X = df[X_cols]
y = df[target_col]

print("X shape:", X.shape)
print("y shape:", y.shape)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("Linear Regression R²:", r2_score(y_test, y_pred_lr))
print("Linear Regression MSE:", mean_squared_error(y_test, y_pred_lr))

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

y_pred_dl = model.predict(X_test).flatten()
print("Deep Learning MSE:", mean_squared_error(y_test, y_pred_dl))

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

plt.figure(figsize=(10, 8))
sns.heatmap(df[X_cols + [target_col]].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()
