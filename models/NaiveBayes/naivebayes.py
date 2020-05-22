import pandas as pd
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import itertools

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
labels = [] 
for df in [no_kernels, lv1_kernels, lv2_kernels]:
    # print(df.head())
    x_df = df.drop('label', axis=1).drop('file_name', axis=1)
    y_df = df['label']
    L = y_df.unique()
    labels.append(L)
    x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.33)
    GNB = GaussianNB().fit(x_train, y_train)
    BNB = BernoulliNB().fit(x_train, y_train)
    gaussian_predictions = GNB.predict(x_test)
    bernoulli_predictions = BNB.predict(x_test)
    g_matrices.append(confusion_matrix(gaussian_predictions, y_test, L,normalize='true'))
    b_matrices.append(confusion_matrix(bernoulli_predictions, y_test, L,normalize='true'))
    g_scores.append((int)(100*GNB.score(x_test,y_test))/100.0)
    b_scores.append((int)(100*BNB.score(x_test,y_test))/100.0)
    
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    import matplotlib.colors as colors
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

f,axes = plt.subplots(3,2)
# plt.title(f'Classes: {labels[0]}')
cmap = truncate_colormap(plt.get_cmap('Blues'), maxval=0.4)

for i in range(3):
    axes[i][0].set_title(f'Gaussian/{PLT_TITLES[i]}/{g_scores[i]}')
    axes[i][1].set_title(f'Bernoulli/{PLT_TITLES[i]}/{b_scores[i]}')
    
    # plot cm as image
    axes[i][0].imshow(g_matrices[i], interpolation='nearest', aspect='auto',cmap=cmap)
    axes[i][0].set_xticks(np.arange(len(labels[i])))
    axes[i][0].set_xticklabels(labels[i])
    axes[i][0].set_yticks(np.arange(len(labels[i])))
    axes[i][0].set_yticklabels(labels[i])
    axes[i][1].imshow(b_matrices[i], interpolation='nearest',aspect='auto',cmap=cmap)
    axes[i][1].set_xticks(np.arange(len(labels[i])))
    axes[i][1].set_xticklabels(labels[i])
    axes[i][1].set_yticks(np.arange(len(labels[i])))
    axes[i][1].set_yticklabels(labels[i])
    
    # add cell labels
    for x, y in itertools.product(range(g_matrices[i].shape[0]), range(g_matrices[i].shape[1])):
        axes[i][0].text(
            y, x, "{:0.2f}".format(g_matrices[i][x, y]),
            horizontalalignment="center",
            verticalalignment="center",
            color="black")
        axes[i][1].text(
            y, x, "{:0.2f}".format(b_matrices[i][x, y]),
            horizontalalignment="center",
            verticalalignment="center",
            color="black")
    
f.tight_layout()
plt.savefig('./models/NaiveBayes/kernel_confusion.png')
