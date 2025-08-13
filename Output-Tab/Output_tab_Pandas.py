import pandas as pd
import numpy as np

# -------------------------
# Create DataFrame from dictionary
# -------------------------
data1 = {'X': [78, 85, 96, 80, 86],
         'Y': [84, 94, 89, 83, 86],
         'Z': [86, 97, 96, 72, 83]}
df1 = pd.DataFrame(data1)
print(df1)

# -------------------------
# Create with index labels
# -------------------------
exam_data = {
    'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
    'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
    'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
    'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']
}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
df2 = pd.DataFrame(exam_data, index=labels)
print(df2)

# Basic summary info
print(df2.info())

# Question 2.2 - First 3 rows
print(df2.head(3))

# 'name' and 'score' columns
print(df2[['name', 'score']])

# Specific rows & columns
print(df2.loc[['b', 'd', 'f', 'g'], ['name', 'score']])

# Attempts > 2
print(df2[df2['attempts'] > 2])

# Shape
print(f"Rows: {df2.shape[0]}, Columns: {df2.shape[1]}")

# Score between 15 and 20
print(df2[df2['score'].between(15, 20)])

# Attempts < 2 and score > 15
print(df2[(df2['attempts'] < 2) & (df2['score'] > 15)])

# Change score in row 'd'
df2.loc['d', 'score'] = 11.5
print(df2)

# Mean of scores
print(df2['score'].mean())

# Append & delete row
df2.loc['k'] = ['Suleman', 18, 2, 'yes']
print("After append", df2)
df2 = df2.drop('k')
print("After delete", df2)

# Sort by name desc, score asc
print(df2.sort_values(by=['name', 'score'], ascending=[False, True]))

# Replace yes/no with True/False
df2['qualify'] = df2['qualify'].map({'yes': True, 'no': False})
print(df2)

# Change 'James' to 'Suresh'
df2['name'] = df2['name'].replace('James', 'Suresh')
print(df2)

# Delete 'attempts' column
df2 = df2.drop(columns='attempts')
print(df2)

# Write to CSV (tab-separated)
df2.to_csv("output_tab.csv", sep='\t', index=True)
print("File saved as 'output_tab.csv' (tab-separated)")
