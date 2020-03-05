# Extraction Modules:
# from components.parse_files.parser import Parser
# from components.utils.test import SimpleTest

# Summaries Modules:
from components.column_summaries.features import Features
from components.extended_summaries.extended_summary import ExtendedSummary
from components.cell_labeling.cell_compact import ContentType

# Clustering Modules
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture
# from sklearn.cluster import DBScan

# Evaluation Modules
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import silhouette_score

# Testing Utilities
from testing.testing_utils import *

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
from glob import glob

CLUSTER_OPTIONS=['kmeans','gmm']

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

args = parser.parse_args()

# Run Column Extraction.
# Test for valid filenames
filenames = [f'data_corpus/{fname}.xlsx' for fname in args.filenames]
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
    Cluster = MiniBatchKMeans
    CLUSTER_TYPE='KMeans_Clustering'
elif args.cluster == 'gmm':
    Cluster = GaussianMixture
    CLUSTER_TYPE='EM_Clustering'
else:
    print('Invalid cluster choice')
    assert args.cluster in CLUSTER_CHOICES

# Scale data using Z-norm
columns_scaled = StandardScaler().fit_transform(columns_vectorized)

# clusters = Cluster.fit_predict(columns_scaled)
# cluster_set, label_set = split_on_cluster(columns_scaled, clusters, column_names)
K_set = np.arange()

