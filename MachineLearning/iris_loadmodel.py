import pickle
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from sklearn import metrics # calculate accuracy

iris = datasets.load_iris()
x = iris.data
y = iris.target

model = pickle.load(open('Iris_classifier.sav','rb'))
y_pred = model.predict(x)

acc = metrics.accuracy_score(y, y_pred)
print(acc)