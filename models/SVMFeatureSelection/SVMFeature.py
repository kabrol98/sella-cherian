from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target
import numpy as np
from matplotlib import pyplot
from xgboost import plot_importance
from numpy import loadtxt
from numpy import sort
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel

alldata = []
with open('data.txt', 'r') as f:
    for line in f:
        data = []
        for word in line.split():
            print(word)
            if ':' not in word:
                data.append(word)
        alldata.append(data)
print(alldata)

#feature for similarity vecotor of a pair of columns
colnames = ["id", "type", "unique", "null", "max", "min", "mean", "std", "common0", "common1", "row_count"]
index = [   0,      1,      2,        3,      4,    5,      6,      7,      8,          9,          10]

#11 col features
'''
0,5,6,7,8,9 important features
id, min, mean, std, common0, common1 #accuracy : 98.4 
'''

label = []
for line in alldata:
    label.append(line[0])
feature = alldata.copy()
for line in feature:
    line.pop(0)
    line.pop(2)
    line.pop(3)
    line.pop(4)
    line.pop(5)
model = XGBClassifier()
feature = np.array(feature)
label = np.array(label)
label = np.ravel(label)
print('ravle label size', np.shape(label))
model.fit(feature, label)
print(model.feature_importances_)
plot_importance(model)
pyplot.show()
f= open("newSVMtraindata.txt", "w+")
alldata = np.concatenate((label.T, feature), axis=1)
f.write(alldata)
X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.25)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
thresholds = sort(model.feature_importances_)
print("sorted features", thresholds)
for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_X_train = selection.transform(X_train)
    selection_model = XGBClassifier()
    selection_model.fit(select_X_train, y_train)
    select_X_test = selection.transform(X_test)
    y_pred = selection_model.predict(select_X_test)
    predictions = y_pred
    accuracy = accuracy_score(y_test, predictions)
    print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))