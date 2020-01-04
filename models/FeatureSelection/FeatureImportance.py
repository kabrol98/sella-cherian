from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target
import pandas
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from matplotlib import pyplot
from xgboost import plot_importance
from numpy import loadtxt
from numpy import sort
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel

colnames = ["file_name","is_alpha", "text_in_header", "is_num","is_alphanum","is_blank", "is_nullDefault", "all_small","all_capital","starts_capital", "contain_colon", "contain_special","text_length","year_range", "has_merge_cell", "right_align","left_align","center_align", "italics_font", "underline_font","bold_font","left_alpha", "left_in_header", "left_num","left_alphanum","left_blank", "above_alpha", "above_in_header","above_num","above_alphanum", "above_blank", "below_alpha","below_in_header","below_num", "below_alphanum", "below_blank","right_alpha","right_in_header", "right_num", "right_alphanum","right_blank","label"]

data1 = pandas.read_csv("newTraining.data", names=colnames, header=None, delimiter=r"\s+")
data2 = pandas.read_csv("newTraining_V2.data", names=colnames, header=None, delimiter=r"\s+")

print(data1.dtypes)

df1 = pandas.DataFrame(data1)
df2 = pandas.DataFrame(data2)
df = pandas.concat([df1, df2], ignore_index=True, sort=False)
features = df[["is_alpha", "text_in_header", "is_num","is_alphanum","is_blank", "is_nullDefault", "all_small","all_capital","starts_capital", "contain_colon", "contain_special","text_length","year_range", "has_merge_cell", "right_align","left_align","center_align", "italics_font", "underline_font","bold_font","left_alpha", "left_in_header", "left_num","left_alphanum","left_blank", "above_alpha", "above_in_header","above_num","above_alphanum", "above_blank", "below_alpha","below_in_header","below_num", "below_alphanum", "below_blank","right_alpha","right_in_header", "right_num", "right_alphanum","right_blank"]]
print("feature shape", features.shape)
label = df[["label"]]
label = np.ravel(label)
print("label shape", label.shape)
encoder = LabelBinarizer()
model = XGBClassifier()
model.fit(features, label)
print(model.feature_importances_)
plot_importance(model)
pyplot.show()

X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.25)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
thresholds = sort(model.feature_importances_)
print("sorted features", thresholds)
for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_X_train = selection.transform(X_train)
    # train model
    selection_model = XGBClassifier()
    selection_model.fit(select_X_train, y_train)
    # eval model
    select_X_test = selection.transform(X_test)
    y_pred = selection_model.predict(select_X_test)
    predictions = y_pred
    accuracy = accuracy_score(y_test, predictions)
    print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))