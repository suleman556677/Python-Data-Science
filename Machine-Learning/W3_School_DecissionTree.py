import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# Create the dataset as a dictionary (since no CSV is provided)
data = {
    'Age': [36, 42, 23, 52, 43, 44, 66, 35, 52, 35, 24, 18, 45],
    'Experience': [10, 12, 4, 4, 21, 14, 3, 14, 13, 5, 3, 3, 9],
    'Rank': [9, 4, 6, 4, 8, 5, 7, 9, 7, 9, 5, 7, 9],
    'Nationality': ['UK', 'USA', 'N', 'USA', 'USA', 'UK', 'N', 'UK', 'N', 'N', 'USA', 'UK', 'UK'],
    'Go': ['NO', 'NO', 'NO', 'NO', 'YES', 'NO', 'YES', 'YES', 'YES', 'YES', 'NO', 'YES', 'YES']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Convert non-numerical columns to numerical
d_nationality = {'UK': 0, 'USA': 1, 'N': 2}
d_go = {'YES': 1, 'NO': 0}
df['Nationality'] = df['Nationality'].map(d_nationality)
df['Go'] = df['Go'].map(d_go)

# Define features and target
features = ['Age', 'Experience', 'Rank', 'Nationality']
X = df[features]
y = df['Go']

# Create and train the Decision Tree Classifier
dtree = DecisionTreeClassifier()
dtree.fit(X, y)

# Visualize the Decision Tree
plt.figure(figsize=(12, 8))
tree.plot_tree(dtree, feature_names=features, class_names=['NO', 'YES'], filled=True)
plt.title("Decision Tree for Comedy Show Attendance")
plt.show()

# Make predictions
# Fix for prediction1
input1 = pd.DataFrame([[40, 10, 7, 1]], columns=features)
prediction1 = dtree.predict(input1)
print("Prediction for Age=40, Experience=10, Rank=7, Nationality=USA:", "YES" if prediction1[0] == 1 else "NO")

# Fix for prediction2
input2 = pd.DataFrame([[40, 10, 6, 1]], columns=features)
prediction2 = dtree.predict(input2)
print("Prediction for Age=40, Experience=10, Rank=6, Nationality=USA:", "YES" if prediction2[0] == 1 else "NO")