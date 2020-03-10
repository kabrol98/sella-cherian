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
# filegroup.add_argument('-f', '--filenames', default=['plasmidsDB'], nargs="*", help='Specify Excel spreadsheet name in data_corpus directory (Omit .xlsx)')
filegroup.add_argument('-S', '--file_sample', default=5, type=float, help="Pick number of files to randomly sample")
# Configure summary type, data type, cluster type.
parser.add_argument('-s', '--summary', default='extended', choices=['standard', 'extended'], help='Choose column summary type.')
parser.add_argument('-d', '--data', default='numeric', choices=['numeric', 'text'], help='Choose between numerical and text data.')
parser.add_argument('-c', '--cluster', default='kmeans', choices=CLUSTER_OPTIONS, help='Choose clustering method')
parser.add_argument('-V', '--vary', help=f'''
                    Choose varying parameter.
                    For reference: {CLUSTER_PARAMS}''')
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
# Filter columns by data type.
if args.data=='numeric':
    DATA_TYPE = 'numeric_data'
    type_class = ContentType.NUMERIC
else:
    DATA_TYPE = 'text_data'
    type_class = ContentType.STRING

columns_filtered = np.extract([x.type==type_class for x in results], results)

# Generate Column Summary Vectors.
if args.summary=='standard':
    SummaryClass = Features
    SUMMARY_TYPE = 'standard_summary'
else:
    SummaryClass = ExtendedSummary
    SUMMARY_TYPE = 'extended_summary'

columns_summary = [
    SummaryClass(c) for c in columns_filtered
]
columns_vectorized = np.array([c.vector for c in columns_summary])
column_names = [c.header for c in columns_summary]
# print(columns_vectorized[0])
N = len(columns_vectorized)

# TODO: Integrate Clustering Modules.
if args.cluster == 'kmeans':
    Cluster = MiniBatchKMeans(n_clusters=int(sqrt(N)), max_iter=100)
    CLUSTER_TYPE='KMEANS'
    cluster_test=test_KMeans
elif args.cluster == 'gmm':
    Cluster = GaussianMixture(n_components=int(sqrt(N)))
    CLUSTER_TYPE='EM'
    cluster_test=test_EM
elif args.cluster == 'dbscan':
    Cluster = DBSCAN(eps=3, min_samples=2)
    CLUSTER_TYPE='DBSCAN'
    cluster_test=test_DBSCAN
elif args.cluster == 'optics':
    Cluster = OPTICS(min_samples=2)
    CLUSTER_TYPE='OPTICS'
    cluster_test=test_OPTICS
elif args.cluster=='none':
    Cluster = NoCluster()
    CLUSTER_TYPE='No_Clusters'
else:
    print('Invalid cluster choice')
    assert args.cluster in CLUSTER_CHOICES

# Scale data using Z-norm
columns_scaled = StandardScaler().fit_transform(columns_vectorized)

# parse param arguments
params = CLUSTER_PARAMS[CLUSTER_TYPE]
pkey = args.vary
if pkey == None:
    pkey = list(params.keys())[0]
    print(f'varying pkey {pkey}')
if pkey not in params:
    print('Error: vary the correct variable.')
    assert pkey in params

# clusters = Cluster.fit_predict(columns_scaled)
# cluster_set, label_set = split_on_cluster(columns_scaled, clusters, column_names)
rng_max = int(sqrt(columns_scaled.shape[0]))
rng_min = 2 if CLUSTER_TYPE in ['KMEANS','EM'] else 1
rng = np.linspace(rng_min,rng_max, num = min(rng_max-1, 20), dtype=np.int32)
# rng = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
clusters = cluster_test(columns_scaled, pkey, params, rng)
columns=columns_scaled
N = columns_scaled.shape[0]
record = []
for index, cluster in enumerate(clusters):
    if len(cluster) == 0 or len(collections.Counter(cluster)) == 1:
        clusters = clusters[:index] + clusters[index+1:]
    else:
        record.append(rng[index])
rng = record
print(rng, clusters)
if len(clusters) != 0:
    db_scores = [davies_bouldin_score(columns, cluster) for cluster in clusters]
    ch_scores = [calinski_harabasz_score(columns, cluster) for cluster in clusters]
    s_scores = [silhouette_score(columns, cluster) for cluster in clusters]
    cluster_nums = [np.max(c)-np.min(c)+1 for c in clusters]
    scores = [db_scores,ch_scores,s_scores, cluster_nums]
    kvals = rng
    scorenames = ['davies_bouldin_score', 'calinski_harabasz_score', 'silhouette_score','Number of clusters']
    f, axes = plt.subplots(4,1)
    for i in range(4):
        ax = axes[i]
        ax = plot_scores(kvals, scores[i], scorenames[i],ax)
    plot_title=f'{N}_columns|{DATA_TYPE}|{SUMMARY_TYPE}|{CLUSTER_TYPE}|{pkey}'
    path_name=f'{N}_columns-{DATA_TYPE}-{SUMMARY_TYPE}-{CLUSTER_TYPE}-{pkey}'
    plt.suptitle(plot_title)
    f.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'testing/cluster_results/{path_name}.png')
    print(f'Saved figure {plot_title} to {path_name}')
else:
    print("there are no valid clusters")
