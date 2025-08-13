import pandas as pd
import numpy as np

# --------------------------
# Load CSV file into DataFrame
# --------------------------
df = pd.read_csv("macro_monthly.csv")

print("=== DataFrame Loaded ===")
print(df)

# --------------------------
# info(), .dtypes, .describe(), .shape
# --------------------------
print("=== DataFrame Info ===")
print(df.info())

print("Data Types:")
print(df.dtypes)

print("Describe:")
print(df.describe())

print("Shape (rows, columns):")
print(df.shape)

# --------------------------
# to_string() examples
# --------------------------
print("=== to_string Examples ===")
print(df.to_string(max_rows=5))  # Limiting rows
print("With selected columns only:")
print(df.to_string(columns=['Year', 'Industrial_Production'], max_rows=5))
print("With custom NA representation and column spacing:")
print(df.to_string(na_rep='MISSING', col_space=15, max_rows=5))

# --------------------------
# Top 4 rows
# --------------------------
print("=== Top 4 Rows ===")
print(df.head(4))

# --------------------------
# Bottom 4 rows
# --------------------------
print("=== Bottom 4 Rows ===")
print(df.tail(4))

# --------------------------
# Access single columns
# --------------------------
print("=== Industrial_Production Column ===")
print(df['Industrial_Production'])

print("Manufacturers_New_Orders: Durable Goods Column:")
print(df['Manufacturers_New_Orders: Durable Goods'])

# --------------------------
# Multiple columns
# --------------------------
print("=== Multiple Columns ===")
print(df[['Industrial_Production', 'Manufacturers_New_Orders: Durable Goods']])

# --------------------------
# Single row using .loc (index 3)
# --------------------------
print("=== Row with index 3 ===")
print(df.loc[3])

# --------------------------
# Multiple rows using .loc
# --------------------------
print("=== Rows 3, 5, 7 ===")
print(df.loc[[3, 5, 7]])

# --------------------------
# Slice rows using .loc (5 to 15)
# --------------------------
print("=== Rows 5 to 15 ===")
print(df.loc[5:15])

# --------------------------
# Conditional selection with .loc
# --------------------------
print("=== Conditional Selection ===")
cond = (df['Year'].isin([1993, 1994, 1997])) & (df['Unemployment_Rate'] >= 1)
print(df.loc[cond])

# --------------------------
# Select specific columns for a specific row (index 9)
# --------------------------
print("\=== Specific Row & Columns ===")
print(df.loc[9, ['Industrial_Production', 'Retail_Sales',
                'Manufacturers_New_Orders: Durable Goods',
                'Personal_Consumption_Expenditures']])

# --------------------------
# Slice of columns using .loc (Industrial_Production <= 0.5)
# --------------------------
print("=== Industrial_Production <= 0.5 ===")
print(df.loc[df['Industrial_Production'] <= 0.5])

# --------------------------
# Combined row and column selection using .loc
# --------------------------
print("=== Industrial_Production <= 0.5 and Consumer_Price Index > 0.2 ===")
print(df.loc[(df['Industrial_Production'] <= 0.5) & (df['Consumer_Price Index'] > 0.2)])

# --------------------------
# Single row using .iloc (4th row)
# --------------------------
print("=== 4th Row ===")
print(df.iloc[3])

# --------------------------
#  Multiple rows using .iloc
# --------------------------
print("=== Multiple Rows ===")
print(df.iloc[[1, 6, 7, 35, 8]])

# --------------------------
# Slice of rows using .iloc (10 to 23)
# --------------------------
print("=== Rows 10 to 23 ===")
print(df.iloc[9:23])

# --------------------------
# Single column using .iloc (5th column)
# --------------------------
print("=== 5th Column ===")
print(df.iloc[:, 4])

# --------------------------
# Multiple columns using .iloc
# --------------------------
print("=== Multiple Columns ===")
print(df.iloc[:, [1, 2, 7]])

# --------------------------
# Slice of columns using .iloc
# --------------------------
print("=== Columns 2 to 8 ===")
print(df.iloc[:, 1:8])

# --------------------------
# Combined row & column selection using .iloc
# --------------------------
print("=== Rows & Columns Selection ===")
print(df.iloc[[3, 4, 6, 24], [2, 4, 6]])

# --------------------------
# Range of rows & columns using .iloc
# --------------------------
print("=== Range Selection ===")
print(df.iloc[2:35, 2:6])

# --------------------------
# Add a new row
# --------------------------
print("=== Add New Row ===")
new_row = df.iloc[0].copy()
df.loc[len(df)] = new_row
print(df.tail())

# --------------------------
# Delete row index 4
# --------------------------
print("=== Delete Row index 4 ===")
df = df.drop(index=4)
print(df.head())

# --------------------------
# Delete rows 5 to 9
# --------------------------
print("=== Delete Rows 5 to 9 ===")
df = df.drop(index=range(5, 10))
print(df.head(15))

# --------------------------
# Delete column All_Employees
# --------------------------
print("=== Delete Column All_Employees ===")
if 'All_Employees' in df.columns:
    df = df.drop(columns=['All_Employees'])
    print("Column 'All_Employees' deleted successfully.")
else:
    print("Column 'All_Employees' not found. Skipping deletion.")
print(df.head())


# --------------------------
# Delete multiple columns
# --------------------------
print("=== Delete Multiple Columns ===")
df = df.drop(columns=['Personal_Consumption_Expenditures', 'National_Home_Price_Index'])
print(df.head())

# --------------------------
# Rename column
# --------------------------
print("=== Rename Column ===")
df.rename(columns={'Personal_Consumption_Expenditures': 'Personal_Consumption_Expenditures_Changed'}, inplace=True)
print(df.head())

# --------------------------
# Rename index label 5 to 8
# --------------------------
print("=== Rename Index Label ===")
df.rename(index={5: 8}, inplace=True)
print(df.head(10))

# --------------------------
# query()
# --------------------------
print("=== query() Example ===")
q_result = df.query('`Industrial_Production` <= 0.5 and `Consumer_Price Index` > 0.2 and Year == 1992')
print(q_result)

# --------------------------
# Sort DataFrame
# --------------------------
print("=== Sort by Consumer_Price Index ===")
print(df.sort_values(by='Consumer_Price Index', ascending=True))

# --------------------------
# Group by Year
# --------------------------
print("=== Group by Year and Sum National_Home_Price_Index ===")
if 'National_Home_Price_Index' in df.columns:
    print(df.groupby('Year')['National_Home_Price_Index'].sum())

# --------------------------
# dropna()
# --------------------------
print("===  Drop rows with NaN ===")
print(df.dropna())

# --------------------------
# Fill NaN with 0
# --------------------------
print("=== Fill NaN with 0 ===")
print(df.fillna(0))
