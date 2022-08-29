import numpy as np
import matplotlib.pyplot as plt

# Adaboost library
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets #Iris dataset
from sklearn.model_selection import train_test_split
from sklearn import metrics # calculate accuracy

import pickle # import/export model

# load datasets
iris = datasets.load_iris()
x = iris.data
y = iris.target

# print(type(x))
# print(type(y))
# print(x.shape)
# print(y.shape)

#split data into training data and test data (30% test, 70% training )
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3)

#create Adaboost classifier
ada = AdaBoostClassifier(n_estimators=30, learning_rate=1)

# train the classifier
model = ada.fit(X_train, Y_train)

#use model
Y_pred = model.predict(X_test)
print('accuracy = ', metrics.accuracy_score(Y_test, Y_pred))

# save model 
filename = 'Iris_classifier.sav'
pickle.dump(model, open(filename, 'wb'))
