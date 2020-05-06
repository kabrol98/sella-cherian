import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree

PLT_TITLES = ['No Kernels', 'Lv1 Kernels', 'Lv2 Kernels']

FEATURE_NAMES = [
    "file_name","is_alpha", "text_in_header", "is_num","is_alphanum",
    "is_blank", "is_nullDefault", "all_small","all_capital","starts_capital",
    "contain_colon", "contain_special","text_length","year_range", "has_merge_cell",
    "right_align","left_align","center_align", "italics_font", "underline_font","bold_font",
    "left_alpha", "left_in_header", "left_num","left_alphanum","left_blank", "above_alpha",
    "above_in_header","above_num","above_alphanum", "above_blank", "below_alpha","below_in_header",
    "below_num", "below_alphanum", "below_blank","right_alpha","right_in_header", "right_num",
    "right_alphanum","right_blank","label"]

no_kernels = pd.read_csv('data_corpus/training_files/NewTraining.data', header=None, sep="\s+")
lv1_kernels = pd.read_csv('data_corpus/training_files/lv1_kernels.csv', header=0)
lv2_kernels = pd.read_csv('data_corpus/training_files/lv2_kernels.csv', header=0)
no_kernels.columns = FEATURE_NAMES

tree_matrices = []
tree_scores = []
for df in [no_kernels, lv1_kernels, lv2_kernels]:
    x_df = df.drop('label', axis=1).drop('file_name', axis=1)
    y_df = df['label']

    x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.33)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    #clf.predict_proba(x_test)

    tree_matrices.append(confusion_matrix(y_pred, y_test))
    tree_scores.append((int)(100*clf.score(x_test,y_test))/100.0)

    #tree.plot_tree(clf.fit(x_test, y_test))

f,axes = plt.subplots(3,1, squeeze=False)
for i in range(3):
    axes[i][0].set_title(f'Decision Tree/{PLT_TITLES[i]}/{tree_scores[i]}')
    axes[i][0].table(tree_matrices[i],loc='center')
    axes[i][0].get_xaxis().set_visible(False)
    axes[i][0].get_yaxis().set_visible(False)

f.tight_layout()
plt.savefig('tree_kernel_confusion.png')
