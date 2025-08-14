import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# CSV Load
df = pd.read_csv("Car Specifications.csv")
print(df)

# DataFrame methods
print("--- INFO ---")
print(df.info())

print("--- DTYPES ---")
print(df.dtypes)

print("--- DESCRIBE ---")
print(df.describe())

print("--- SHAPE ---")
print(df.shape)

# Select independent & dependent variables
X = df[['cylinders', 'displacement', 'weight']]
y = df['mpg'] 

X_array = X.values
y_array = y.values

# Regression plots
for col in X.columns:
    sns.regplot(x=df[col], y=y, ci=None, line_kws={"color": "red"})
    plt.title(f"Regression Line: {col} vs MPG")
    plt.show()

# Correlation heatmap
corr = df[['cylinders', 'displacement', 'weight', 'mpg']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Train-test split (90% train, 10% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_array, y_array, test_size=0.1, random_state=42
)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Intercept
print("Intercept:", model.intercept_)

# Slopes
print("Slopes:", model.coef_)

# Predict MPG for test data
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
