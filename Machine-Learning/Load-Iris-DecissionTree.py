from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import pandas as pd

# Load the Iris dataset
iris = load_iris()

# Convert to DataFrame for clarity
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Create and train the model
X = df[iris.feature_names]
y = df['target']
clf = DecisionTreeClassifier()
clf.fit(X, y)

# Visualize the tree
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Decision Tree for Iris Classification")
plt.show()

# Make a prediction
sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=iris.feature_names)
pred = clf.predict(sample)
print("Predicted class:", iris.target_names[pred[0]])