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

# Silimarity Modules
from components.similarity.cosine_similarity import CosineSimilarity
# TODO: Add similarity modules for testing purposes.

# Testing Utilities
from testing.testing_utils import *
from testing.cluster_utils import *

# Python Modules
from enum import Enum
import numpy as np
import keras
import tensorflow as tf
# import keras.backend as K
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
parser.add_argument('-V', '--vary', help=f'''
                    Choose varying parameter.
                    For reference: {CLUSTER_PARAMS}''')
args = parser.parse_args()
# print(args)
# Run Column Extraction.
args = parse_args()
# Run Column Extraction.
# Test for valid filenames
filenames = sample_dataset(args.file_sample, args.filenames)

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
N = len(columns_vectorized)

# TODO: Integrate Clustering Modules.
if args.cluster == 'kmeans':
    Cluster = MiniBatchKMeans(n_clusters=int(sqrt(N)), max_iter=100)
    CLUSTER_TYPE='KMeans'
elif args.cluster == 'gmm':
    Cluster = GaussianMixture(n_components=int(sqrt(N)))
    CLUSTER_TYPE='EM'
elif args.cluster == 'dbscan':
    Cluster = DBSCAN(eps=3, min_samples=2)
    CLUSTER_TYPE='DBSCAN'
elif args.cluster == 'optics':
    Cluster = OPTICS(min_samples=2)
    CLUSTER_TYPE='OPTICS'
else:
    Cluster = NoCluster()
    CLUSTER_TYPE='No_Clusters'
    

# Scale data using Z-norm
columns_scaled = StandardScaler().fit_transform(columns_vectorized)

# add scaled testing
canalyse_path = f'{args.filename}-{SUMMARY_TYPE}-{DATA_TYPE}'
if args.canalyse:
    pickle.dump(columns_scaled, open(f'testing/canalyse/{canalyse_path}.p', "wb"))

clusters = Cluster.fit_predict(columns_scaled)
cluster_set, label_set = split_on_cluster(columns_scaled, clusters, column_names)
# Filter empty clusters.
nz_filter = list(map(np.any, cluster_set))
clusters_nonzero = cluster_set[nz_filter]
labels_nonzero = label_set[nz_filter]

# Run Similarity Analysis
# TODO: Implement alternate similarity modules.
SimilarityClass = CosineSimilarity

cosine_set = SimilarityClass(clusters_nonzero).cosine_set

save_path = f'{SUMMARY_TYPE}-{DATA_TYPE}-{CLUSTER_TYPE}-{KEY_VARY}'
plot_title = f'{SUMMARY_TYPE}||{DATA_TYPE}||{CLUSTER_TYPE}'

# Plot results.
plot_results(
    cosine_set,
    labels_nonzero,
    plot_title,
    save_path
)
