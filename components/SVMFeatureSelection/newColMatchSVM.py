from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn import metrics
alldata = []
with open('newSVMdata.txt','r') as f:
    for line in f:
        data = []
        for word in line.split():
            data.append(float(word))
        alldata.append(data)
print(alldata)


label = []
for line in alldata:
    label.append(line[0])


feature = alldata.copy()
for line in feature:
    line.pop(0)

X = np.array(feature)
Y = np.array(label)
print(type(X))
print(type(Y))

print(X)
print(Y)
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)


logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
y_pred_class = logreg.predict(X_test)
print('logregress accuracy', metrics.accuracy_score(Y_test, y_pred_class))
print('logregress f1', metrics.f1_score(Y_test, y_pred_class))
print('logregress precision score', precision_score(Y_test, y_pred_class))
print('logregress recall', metrics.recall_score(Y_test, y_pred_class))

svm = svm.SVC(gamma='scale')
svm.fit(X_train, Y_train)
y_pred_class = svm.predict(X_test)
print('svm accuracy', metrics.accuracy_score(Y_test, y_pred_class))
print('svm f1', metrics.f1_score(Y_test, y_pred_class))
print('svm precision score', precision_score(Y_test, y_pred_class))
print('svm recall', metrics.recall_score(Y_test, y_pred_class))

NB = GaussianNB()
NB.fit(X_train, Y_train)
y_pred_class = NB.predict(X_test)
print('NB accuracy', metrics.accuracy_score(Y_test, y_pred_class))
print('NB f1', metrics.f1_score(Y_test, y_pred_class))
print('NB precision score', precision_score(Y_test, y_pred_class))
print('NB recall', metrics.recall_score(Y_test, y_pred_class))

'''
logregress accuracy 0.9698795180722891
logregress f1 0.8484848484848485
logregress precision score 0.7777777777777778
logregress recall 0.9333333333333333

svm accuracy 0.9819277108433735
svm f1 0.9032258064516129
svm precision score 0.875
svm recall 0.9333333333333333

NB accuracy 0.9156626506024096
NB f1 0.6818181818181819
NB precision score 0.5172413793103449
NB recall 1.0
'''

