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
from sklearn.cluster import OPTICS

import openpyxl

# Silimarity Modules
from components.similarity.cosine_similarity import CosineSimilarity
# TODO: Add similarity modules for testing purposes.

# Testing Utilities
from testing.test_utils.testing_utils import *

# Python Modules
from enum import Enum
import numpy as np
import keras
import tensorflow as tf
# import keras.backend as K
import argparse
from os import path
from math import sqrt
import pickle

# Configure argument parser
parser = argparse.ArgumentParser(description='''
                                Tests sella pipeline on given excel spreadsheet.
                                Outputs Confusion matrix of column similarity test
                                 into testing/confusion_results.
                                Use command line arguments to configure different test types.
                                 ''')
parser.add_argument('-f', '--filename', default='plasmidsDB', help='Specify Excel spreadsheet name in data_corpus directory (Omit .xlsx)')
# Configure summary type, data type, cluster type.
parser.add_argument('-s', '--summary', default='extended', choices=['standard', 'extended'], help='Choose column summary type.')
parser.add_argument('-d', '--data', default='numeric', choices=['numeric', 'text'], help='Choose between numerical and text data.')
parser.add_argument('-c', '--cluster', default='none', choices=['none','kmeans','gmm','dbscan', 'optics'], help='Choose clustering method')
parser.add_argument('-A', '--canalyse', default='none', action="store_true", help='Choose clustering method')

args = parser.parse_args()
# print(args)
# Run Column Extraction.

# Test for valid filename
# filename = f'data_corpus/{args.filename}.xlsx'
type_class = ContentType.NUMERIC
SummaryClass = Features
SUMMARY_TYPE = 'standard_summary'

filename = "training_data/row_containment/data/data_651032503351853.xlsx"
filenames = ["training_data/row_containment/data/data_752184637536815.xlsx", "training_data/row_containment/data/data_651032503351853.xlsx", "training_data/row_containment/spreadsheet/11a10.xlsx", ]
size_files = {}

# if not path.exists(filename):
#     print(f'File {filename} does not exist!')
# assert path.exists(filename)

model_path = "models/NeuralNetwork/vertical_lstm.h5"
print(f'Extracting columns from {filename}...')

with SimpleTest():

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    model = keras.models.load_model(model_path)

    results = []
    for file in filenames:
        parser = Parser(file, model)
        for res in parser.parse():
            results.append(res)
    print(f'Extracted {len(results)} columns!')

# Filter columns by data type.
# if args.data=='numeric':
#     DATA_TYPE = 'numeric_data'
#     type_class = ContentType.NUMERIC
# else:
#     DATA_TYPE = 'text_data'
#     type_class = ContentType.STRING

# Generate Column Summary Vectors.
# if args.summary=='standard':
#     SummaryClass = Features
#     SUMMARY_TYPE = 'standard_summary'
# else:
#     SummaryClass = ExtendedSummary
#     SUMMARY_TYPE = 'extended_summary'

columns_filtered = np.extract([x.type==type_class for x in results], results)

columns_summary = [
    SummaryClass(c) for c in columns_filtered
]
columns_vectorized = np.array([c.vector for c in columns_summary])
column_names = [c.column_metadata.file_name for c in columns_summary]

for file_columns in column_names:
    if file_columns not in size_files:
        size_files[file_columns] = 0
    size_files[file_columns] += 1

print(size_files)

# print(columns_vectorized[0])
N = len(columns_vectorized)

Cluster = MiniBatchKMeans(n_clusters=int(sqrt(N)), max_iter=100)
CLUSTER_TYPE='KMeans_Clustering'

# TODO: Integrate Clustering Modules.
# if args.cluster == 'kmeans':
#     Cluster = MiniBatchKMeans(n_clusters=int(sqrt(N)), max_iter=100)
#     CLUSTER_TYPE='KMeans_Clustering'
# elif args.cluster == 'gmm':
#     Cluster = GaussianMixture(n_components=int(sqrt(N)))
#     CLUSTER_TYPE='EM_Clustering'
# elif args.cluster == 'dbscan':
#     Cluster = DBSCAN(eps=3, min_samples=2)
#     CLUSTER_TYPE='DB_Clustering'
# elif args.cluster == 'optics':
#     Cluster = OPTICS(min_samples=2)
#     CLUSTER_TYPE='OP_Clustering'
# else:
#     Cluster = NoCluster()
#     CLUSTER_TYPE='No_Clusters'

# Scale data using Z-norm
columns_scaled = StandardScaler().fit_transform(columns_vectorized)

# add scaled testing
# canalyse_path = f'{args.filename}-{SUMMARY_TYPE}-{DATA_TYPE}'
# if args.canalyse:
#     pickle.dump(columns_scaled, open(f'testing/canalyse/{canalyse_path}.p', "wb"))


clusters = Cluster.fit_predict(columns_scaled)

def split_on_cluster(columns, clusters, column_names):
    cluster_results = {}
    name_results = {}
    for index, element in enumerate(columns):
        if clusters[index] not in cluster_results:
            cluster_results[clusters[index]] = []
            name_results[clusters[index]] = []
        cluster_results[clusters[index]].append(element)
        name_results[clusters[index]].append(column_names[index])
    cluster_set = []
    name_set = []
    for key in cluster_results.keys():
        cluster_set.append(np.array(cluster_results[key]))
        name_set.append(np.array(name_results[key]))
    return np.array(cluster_set), np.array(name_set), cluster_results.keys()

cluster_set, name_set, label_set = split_on_cluster(columns_scaled, clusters, column_names)
print(name_set)

# Filter empty clusters.
# nz_filter = list(map(np.any, cluster_set))
# clusters_nonzero = cluster_set[nz_filter]
# labels_nonzero = label_set[nz_filter]
#
# # Run Similarity Analysis
# # TODO: Implement alternate similarity modules.
SimilarityClass = CosineSimilarity

cosine_set = SimilarityClass(cluster_set).compute_sim()

my_result = {}

for set in range(len(name_set)):
    for i in range(len(name_set[set]) - 1):
        for j in range(i + 1, len(name_set[set])):
            file1 = name_set[set][i]
            file2 = name_set[set][j]
            if file1 == file2:
                continue
            if file2 < file1:
                temp = file2
                file2 = file1
                file1 = temp
            if file1 + "__" + file2 not in my_result:
                my_result[file1 + "__" + file2] = 0
            my_result[file1 + "__" + file2] += cosine_set[set][i][j] / (size_files[file1] * size_files[file2])

print(my_result)
# # exit()
# save_path = f'{args.filename}-{SUMMARY_TYPE}-{DATA_TYPE}-{CLUSTER_TYPE}'
# plot_title = f'{args.filename}||{SUMMARY_TYPE}||{DATA_TYPE}||{CLUSTER_TYPE}'
# # Plot results.

# save_path = "yitao.png"
# plot_title = "hi"
#
# plot_pipeline(
#     cosine_set,
#     list(label_set),
#     plot_title,
#     save_path
# )
