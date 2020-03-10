# Extraction Modules:
from components.parse_files.parser import Parser
from components.utils.test import SimpleTest

# Summaries Modules:
from components.column_summaries.features import Features
from components.extended_summaries.extended_summary import ExtendedSummary
from components.cell_labeling.cell_compact import ContentType

# Clustering Modules
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN

# Evaluation Modules
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import silhouette_score
from sklearn.cluster import OPTICS

# Testing Utilities
from testing.testing_utils import *
from testing.cluster_utils import *
from testing.plot_utils import *

# Python Modules
from enum import Enum
import numpy as np
import keras
import tensorflow as tf
import keras.backend as K
import argparse
from os import path
from math import sqrt
import pickle
from glob import glob
from matplotlib import pyplot as plt
import collections

CLUSTER_OPTIONS=['kmeans','gmm', 'dbscan', 'optics','birch']

# Configure argument parser
parser = argparse.ArgumentParser(description='''
                                Tests sella pipeline on given excel spreadsheet.
                                Outputs Confusion matrix of column similarity test
                                 into testing/confusion_results.
                                Use command line arguments to configure different test types.
                                 ''')
filegroup = parser.add_mutually_exclusive_group()
filegroup.add_argument('-S', '--file_sample', default=5, type=float, help="Pick number of files to randomly sample")

args = parser.parse_args()

# Run Column Extraction.
# Test for valid filenames
filenames = sample_dataset(args.file_sample, None)
NUM_FILES=len(filenames)
for n in filenames:
    if not path.exists(n):
        print(f'File {n} does not exist!')
    assert path.exists(n)

model_path = "models/NeuralNetwork/vertical_lstm.h5"
print(f'Extracting columns from dataset...')

results = []
for filename in filenames:
    with SimpleTest():

        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        model = keras.models.load_model(model_path)

        parser = Parser(filename, model)
        for res in parser.parse():
            results.append(res)
        print(f'Extracted {len(results)} columns from {filename}...')

DATA_TYPE = 'numeric_data'
type_class = ContentType.NUMERIC

columns_filtered = np.extract([x.type==type_class for x in results], results)

# Generate Column Summary Vectors.
SummaryClass = ExtendedSummary
SUMMARY_TYPE = 'extended_summary'

columns_summary = [
    SummaryClass(c) for c in columns_filtered
]
columns_vectorized = np.array([c.vector for c in columns_summary])
column_names = [c.header for c in columns_summary]
# print(columns_vectorized[0])
N = len(columns_vectorized)

# Scale data using Z-norm
columns_scaled = StandardScaler().fit_transform(columns_vectorized)

# rng = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
cluster_name = ["kmeans", "em", "dbscan", "optics"]
cluster_methods = [PLOT_KMeans, PLOT_EM, PLOT_DBSCAN, PLOT_OPTICS]

file = open("result.txt", "a")
file.write(str(N))
file.write("\n")

for index, cluster_test in enumerate(cluster_methods):
    clusters = cluster_test(columns_scaled)
    columns=columns_scaled
    N = columns_scaled.shape[0]
    db_scores = str(davies_bouldin_score(columns, clusters))
    ch_scores = str(calinski_harabasz_score(columns, clusters))
    s_scores = str(silhouette_score(columns, clusters))
    cluster_nums = str(np.max(clusters)-np.min(clusters) + 1)
    file.write(cluster_name[index] + " " + db_scores + " " + ch_scores + " " + s_scores + " " + cluster_nums + "\n")

file.close()
# record = []
# for index, cluster in enumerate(clusters):
#     if len(cluster) == 0 or len(collections.Counter(cluster)) == 1:
#         clusters = clusters[:index] + clusters[index+1:]
#     else:
#         record.append(rng[index])
# rng = record
# print(rng, clusters)
# if len(clusters) != 0:
#     db_scores = [davies_bouldin_score(columns, cluster) for cluster in clusters]
#     ch_scores = [calinski_harabasz_score(columns, cluster) for cluster in clusters]
#     s_scores = [silhouette_score(columns, cluster) for cluster in clusters]
#     cluster_nums = [np.max(c)-np.min(c)+1 for c in clusters]
#     scores = [db_scores,ch_scores,s_scores, cluster_nums]
#     kvals = rng
#     scorenames = ['davies_bouldin_score', 'calinski_harabasz_score', 'silhouette_score','Number of clusters']
#     f, axes = plt.subplots(4,1)
#     for i in range(4):
#         ax = axes[i]
#         ax = plot_scores(kvals, scores[i], scorenames[i],ax)
#     plot_title=f'{N}_columns|{DATA_TYPE}|{SUMMARY_TYPE}|{CLUSTER_TYPE}|{pkey}'
#     path_name=f'{N}_columns-{DATA_TYPE}-{SUMMARY_TYPE}-{CLUSTER_TYPE}-{pkey}'
#     plt.suptitle(plot_title)
#     f.tight_layout(rect=[0, 0.03, 1, 0.95])
#     plt.savefig(f'testing/cluster_results/{path_name}.png')
#     print(f'Saved figure {plot_title} to {path_name}')
# else:
#     print("there are no valid clusters")
