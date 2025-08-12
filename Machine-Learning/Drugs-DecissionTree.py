import pandas as pd 
import sklearn as skl
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv("drug200.csv" , delimiter="," , on_bad_lines="skip" )

print(data)
print(data.info())
print(data.describe())
print(data.shape)

#==============

X =  data[["Age","Sex","BP","Cholesterol","Na_to_K"]]

Y = data[["Drug"]]

X.loc[:, 'Sex'] = X['Sex'].map({'F': 0, 'M': 1})
X.loc[:, 'BP'] = X['BP'].map({'HIGH': 0, 'LOW': 1, 'NORMAL': 2})
X.loc[:, 'Cholesterol'] = X['Cholesterol'].map({'HIGH': 0, 'NORMAL': 1})
Y.loc[:, 'Drug'] = Y['Drug'].map({'drugY': 0, 'drugX': 1, 'drugA': 2, 'drugB':3,'drugC':4})


X.loc[:, 'Age'].fillna(X['Age'].median(), inplace=True)
X.loc[:, 'BP'].fillna(0, inplace=True)
X.loc[:, 'Cholesterol'].fillna(0, inplace=True)
X.loc[:, 'Na_to_K'].fillna(0, inplace=True)
Y.loc[:, 'Drug'].fillna(0, inplace=True)

print(X)
print(Y)

Xi =  X.values.reshape(-1,5)

Yi = Y.values.reshape(-1,1)
Yi=Yi.astype('int')

#X= X[:,1]
#Y=Y[:,1]
Xtrain , XTest,YTrain, YTest =  train_test_split (Xi,Yi, test_size=0.2 )

dst  =  DecisionTreeClassifier()

dst.fit( Xtrain, YTrain)

Pridict_Y = dst.predict(XTest)

print("Predict:")
print(Pridict_Y)

print("Y Test: ")
print(YTest)