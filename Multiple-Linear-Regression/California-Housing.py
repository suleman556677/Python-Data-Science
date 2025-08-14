import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# ----------------------------
# Load CSV
# ----------------------------
df = pd.read_csv("California-Housing.csv")
print("--- First 5 Rows ---")
print(df.head())

print("--- Info ---")
df.info()

print("--- Data Types ---")
print(df.dtypes)

print("--- Shape ---")
print(df.shape)

# ----------------------------
# Handle Categorical Columns
# ----------------------------
if 'ocean_proximity' in df.columns:
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(df[['ocean_proximity']])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['ocean_proximity']))
    df = pd.concat([df.drop('ocean_proximity', axis=1), encoded_df], axis=1)

# ----------------------------
# Target & Features
# ----------------------------
target_col = "median_house_value"
X = df.drop(target_col, axis=1)
y = df[target_col]

# ----------------------------
# Handle Missing Values
# ----------------------------
imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
y = pd.Series(y).fillna(y.median())  # Fill NaNs in y if any

# ----------------------------
# Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

# ----------------------------
# Train Model
# ----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ----------------------------
# Predictions & Metrics
# ----------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("--- Model Metrics ---")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")

# ----------------------------
# Coefficients Table
# ----------------------------
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", key=abs, ascending=False)
print("--- Coefficients ---")
print(coef_df)

# ----------------------------
# Regression Plots (Clean)
# ----------------------------
numeric_cols = X.columns[:6]  # Limit to first 6 for clarity
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    sns.regplot(
        x=df[col], y=df[target_col],
        scatter_kws={"alpha": 0.4, "s": 40}, 
        line_kws={"color": "red"},
        ax=axes[i]
    )
    axes[i].set_title(f"{col} vs {target_col}", fontsize=14)
    axes[i].set_xlabel(col, fontsize=12)
    axes[i].set_ylabel(target_col, fontsize=12)
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ----------------------------
# Correlation Heatmap
# ----------------------------
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", cbar_kws={"shrink": 0.8})
plt.title("Correlation Heatmap", fontsize=16)
plt.show()
