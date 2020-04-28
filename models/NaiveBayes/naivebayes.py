import pandas as pd
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

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


g_matrices = []
b_matrices = []
g_scores = []
b_scores = []
for df in [no_kernels, lv1_kernels, lv2_kernels]:
    # print(df.head())
    x_df = df.drop('label', axis=1).drop('file_name', axis=1)
    y_df = df['label']
    # print(x_df)

    x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.33)
    GNB = GaussianNB().fit(x_train, y_train)
    BNB = BernoulliNB().fit(x_train, y_train)
    gaussian_predictions = GNB.predict(x_test)
    bernoulli_predictions = BNB.predict(x_test)
    g_matrices.append(confusion_matrix(gaussian_predictions, y_test))
    b_matrices.append(confusion_matrix(bernoulli_predictions, y_test))
    g_scores.append((int)(100*GNB.score(x_test,y_test))/100.0)
    b_scores.append((int)(100*BNB.score(x_test,y_test))/100.0)
    
    
f,axes = plt.subplots(3,2)
for i in range(3):
    axes[i][0].set_title(f'Gaussian/{PLT_TITLES[i]}/{g_scores[i]}')
    axes[i][1].set_title(f'Bernoulli/{PLT_TITLES[i]}/{b_scores[i]}')
    axes[i][0].table(g_matrices[i], loc='center')
    axes[i][1].table(b_matrices[i], loc='center')
    axes[i][0].get_xaxis().set_visible(False)
    axes[i][1].get_xaxis().set_visible(False)
    axes[i][0].get_yaxis().set_visible(False)
    axes[i][1].get_yaxis().set_visible(False)
    
f.tight_layout()
plt.show()
    
