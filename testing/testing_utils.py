import matplotlib.pyplot as plt
import numpy as np


class NoCluster:
    def __init__(self):
        return
    def fit_predict(self, X):
        N = X.shape[0]
        return np.ones((N,), dtype=int)

def split_on_cluster(matrix, assignments, labels):
    # TODO:Finish
    K = np.max(assignments) + 1
    # print(assignments)
    cluster_set = np.array([matrix[assignments==i] for i in range(K)])
    label_set = np.array([np.extract([assignments==i], labels) for i in range(K)])
    return cluster_set, label_set

def plot_results(cosine_set: np.array,
                 label_set: np.array,
                 plot_title: np.array,
                 path_name: str):

    # create subplots
    plt.rcParams.update({'font.size': 7})
    N = cosine_set.shape[0]
    num_cols = int(np.ceil(np.sqrt(N)))
    num_rows = int(np.ceil(N/num_cols))
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

