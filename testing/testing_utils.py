import matplotlib.pyplot as plt
import numpy as np
import glob
from sklearn.utils.random import sample_without_replacement as sample
from LocalitySensitiveHashing import LocalitySensitiveHashing
import argparse

def parse_args():
    CLUSTER_OPTIONS=['kmeans','gmm', 'lsh', 'none']
    # Configure argument parser
    parser = argparse.ArgumentParser(description='''
                                    Tests sella pipeline on given excel spreadsheet.
                                    Outputs Confusion matrix of column similarity test
                                    into testing/confusion_results.
                                    Use command line arguments to configure different test types.
                                    ''')
    filegroup = parser.add_mutually_exclusive_group()
    filegroup.add_argument('-f', '--filenames', default=['plasmidsDB'], nargs="*", help='Specify Excel spreadsheet name in data_corpus directory (Omit .xlsx)')
    filegroup.add_argument('-S', '--file_sample', type=float, help="Pick number of files to randomly sample")
    # Configure summary type, data type, cluster type.
    parser.add_argument('-s', '--summary', default='extended', choices=['standard', 'extended'], help='Choose column summary type.')
    parser.add_argument('-d', '--data', default='numeric', choices=['numeric', 'text'], help='Choose between numerical and text data.')
    parser.add_argument('-c', '--cluster', default='none', choices=CLUSTER_OPTIONS, help='Choose clustering method')
    parser.add_argument('-A', '--canalyse', default='none', action="store_true", help='Choose clustering method')

    return parser.parse_args()

def sample_dataset(s, filenames):
    if s == None or s <= 0:
        return [f'data_corpus/{fname}.xlsx' for fname in filenames]
    else:
        filenames = np.array(glob.glob('data_corpus/*.xlsx'))
        n = len(filenames)
        s = min(s,n)
        index = sample(n, s)
        return filenames[index]
    
def plot_results(cosine_set: np.array,
                 label_set: np.array,
                 plot_title: np.array,
                 path_name: str):

    # create subplots
    plt.rcParams.update({'font.size': 7})
    N = cosine_set.shape[0]
    num_cols = int(np.ceil(np.sqrt(N)))
    num_rows = int(np.ceil(np.sqrt(N)))
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    # Enumerate over clusters.
    for it in range(N):
        col = int(np.floor(it / (num_cols+0.0)))
        row = it % num_cols
        # print(f'({row}, {col})({num_rows}, {num_cols}) {N} {it}')
        ax = axes[row,col]
        matrix = cosine_set[it]
        col_names = label_set[it]
        # Label axes
        n = len(col_names)
        rng = np.arange(n)
        ax.set_xticks(rng)
        ax.set_yticks(rng)
        ax.set_yticklabels(col_names)
        # Annotate Similarity Values.
        for i in range(n):
            for j in range(n):
                text = ax.text(j, i, "{0:.2f}".format(matrix[i, j]),
                            ha="center", va="center", color="b")
        sub_title = f'Class {it}'
        ax.set_title(sub_title)
        ax.imshow(matrix, cmap='Pastel1')
    fig.suptitle(plot_title)
    fig.tight_layout(pad=2.5)
    plt.savefig(f'testing/confusion_results/{path_name}.png')
    print(f'Saved figure {plot_title} to {path_name}')

