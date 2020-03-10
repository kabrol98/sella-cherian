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
from copy import deepcopy as cpy
import csv

GLOBAL_COUNT = 0
# Configure argument parser
parser = argparse.ArgumentParser(description='''
                                Tests sella pipeline on given excel spreadsheet.
                                Outputs Confusion matrix of column similarity test
                                 into testing/confusion_results.
                                Use command line arguments to configure different test types.
                                 ''')
filegroup = parser.add_mutually_exclusive_group()
# filegroup.add_argument('-f', '--filenames', default=['plasmidsDB'], nargs="*", help='Specify Excel spreadsheet name in data_corpus directory (Omit .xlsx)')
filegroup.add_argument('-S', '--file_sample', default=1, type=int, help="Pick number of files to randomly sample")
# Configure summary type, data type, cluster type.
# parser.add_argument('-s', '--summary', default='extended', choices=['standard', 'extended'], help='Choose column summary type.')
# parser.add_argument('-d', '--data', default='numeric', choices=['numeric', 'text'], help='Choose between numerical and text data.')
# parser.add_argument('-c', '--cluster', default='kmeans', choices=CLUSTER_OPTIONS, help='Choose clustering method')

args = parser.parse_args()
# print(args).
# Run Column Extraction.
# Test for valid filenames
filenames = sample_dataset(args.file_sample, None)

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
        

numeric_filtered = np.extract([x.type==ContentType.NUMERIC for x in results], results)
text_filtered = np.extract([x.type==ContentType.STRING for x in results], results)

# Generate Column Summary Vectors.
# if args.summary=='standard':
#     SummaryClass = Features
#     SUMMARY_TYPE = 'standard_summary'
# else:
#     SummaryClass = ExtendedSummary
#     SUMMARY_TYPE = 'extended_summary'
SummaryClass=ExtendedSummary

numeric_summary = [
    SummaryClass(c) for c in numeric_filtered
]
numeric_vectorized = np.array([c.vector for c in numeric_summary])
numeric_column_names = [c.colname for c in numeric_summary]
numeric_sheet_names = [c.sheetname for c in numeric_summary]
numeric_file_names = [c.filename for c in numeric_summary]
numeric_id = [c.id for c in numeric_summary]
numeric_N = numeric_vectorized.shape[0]

text_summary = [
    SummaryClass(c) for c in text_filtered
]
text_vectorized = np.array([c.vector for c in text_summary])
# text_column_names = [c.header for c in text_summary]
text_column_names = [c.colname for c in text_summary]
text_sheet_names = [c.sheetname for c in text_summary]
text_file_names = [c.filename for c in text_summary]
text_id = [c.id for c in text_summary]
text_N = text_vectorized.shape[0]
print(numeric_filtered.shape, len(numeric_summary), numeric_vectorized.shape)
# Scale data using Z-norm
numeric_scaled = StandardScaler().fit_transform(numeric_vectorized)
text_scaled = StandardScaler().fit_transform(text_vectorized)

numeric_Cluster = MiniBatchKMeans(n_clusters=int(sqrt(numeric_N)), max_iter=100)   
text_Cluster = MiniBatchKMeans(n_clusters=int(sqrt(text_N)), max_iter=100)   

numeric_clusters = numeric_Cluster.fit_predict(numeric_scaled)
numeric_cluster_set, numeric_label_set, numeric_sheet_set, numeric_file_set, numeric_id_set = split_on_cluster(
    numeric_scaled, 
    numeric_clusters, 
    numeric_column_names, 
    numeric_sheet_names,
    numeric_file_names,
    numeric_id)

text_clusters = text_Cluster.fit_predict(text_scaled)
text_cluster_set, text_label_set, text_sheet_set, text_file_set, text_id_set = split_on_cluster(
    text_scaled, 
    text_clusters, 
    text_column_names, 
    text_sheet_names,
    text_file_names,
    text_id)

# Filter empty clusters.
nz_filter = list(map(np.any, numeric_cluster_set))
numeric_clusters_nonzero = numeric_cluster_set[nz_filter]
numeric_labels_nonzero = numeric_label_set[nz_filter]
numeric_sheets_nonzero = numeric_sheet_set[nz_filter]
numeric_files_nonzero = numeric_file_set[nz_filter]
numeric_id_nonzero = numeric_id_set[nz_filter]
numeric_n_clusters = numeric_clusters_nonzero.shape[0]

nz_filter = list(map(np.any, text_cluster_set))
text_clusters_nonzero = text_cluster_set[nz_filter]
text_labels_nonzero = text_label_set[nz_filter]
text_sheets_nonzero = text_sheet_set[nz_filter]
text_files_nonzero = text_file_set[nz_filter]
text_id_nonzero = text_id_set[nz_filter]
text_n_clusters = text_clusters_nonzero.shape[0]
print("Ran Clusters...")
# Run Similarity Analysis
# TODO: Implement alternate similarity modules.
SimilarityClass = CosineSimilarity

numeric_cosine_set = SimilarityClass(numeric_clusters_nonzero).cosine_set
text_cosine_set = SimilarityClass(text_clusters_nonzero).cosine_set
print("Generated cosine sets")
graph_row = {
    'A_id':'',
    'B_id':'',
    'score':-1
    }
node_row = {
    'id':'','file_name':'','sheet_name':'','column_name':'','data_type':''
}
rel_rows = []
node_rows = []
graph_row['A_data_type'] = 'numeric'
graph_row['B_data_type'] = 'numeric'
node_row['data_type'] = 'numeric'
for c in range(numeric_n_clusters):
    cluster = numeric_clusters_nonzero[c]
    colnames = numeric_labels_nonzero[c]
    sheetnames = numeric_sheets_nonzero[c]
    filenames = numeric_files_nonzero[c]
    ids = numeric_id_nonzero[c]
    cosines = numeric_cosine_set[c]
    n_cols = cluster.shape[0]
    for i in range(n_cols):
        node_row['id'] = ids[i]
        node_row['file_name'] = filenames[i]
        node_row['sheet_name'] = sheetnames[i]
        node_row['column_name'] = colnames[i]
        node_rows.append(cpy(node_row))
        for j in range(i+1,n_cols):
            graph_row['A_id'] = ids[i]
            graph_row['B_id'] = ids[j]
            graph_row['score'] = cosines[i][j]
            rel_rows.append(cpy(graph_row))
            
            graph_row['A_id'] = ids[j]
            graph_row['B_id'] = ids[i]
            graph_row['score'] = cosines[j][i]
            rel_rows.append(cpy(graph_row))

graph_row['A_data_type'] = 'text'
graph_row['B_data_type'] = 'text'
node_row['data_type'] = 'text'
for c in range(text_n_clusters):
    cluster = text_clusters_nonzero[c]
    colnames = text_labels_nonzero[c]
    sheetnames = text_sheets_nonzero[c]
    filenames = text_files_nonzero[c]
    ids = text_id_nonzero[c]
    cosines = text_cosine_set[c]
    n_cols = cluster.shape[0]
    for i in range(n_cols):
        node_row['id'] = ids[i]
        node_row['file_name'] = filenames[i]
        node_row['sheet_name'] = sheetnames[i]
        node_row['column_name'] = colnames[i]
        node_rows.append(cpy(node_row))
        for j in range(i+1,n_cols):
            graph_row['A_id'] = ids[i]
            graph_row['B_id'] = ids[j]
            graph_row['score'] = cosines[i][j]
            rel_rows.append(cpy(graph_row))
            
            graph_row['A_id'] = ids[j]
            graph_row['B_id'] = ids[i]
            graph_row['score'] = cosines[j][i]
            rel_rows.append(cpy(graph_row))

print("Generated graph...")
COUNT = 0
nodes = set()
for row in node_rows:
    if row['id'] in nodes:
        row['ID'] += str(COUNT)
        COUNT += 1
    nodes.add(row['id'])

nodes_path = f'testing/graphs/{args.file_sample}_samples-{numeric_N}_numeric-{text_N}_text-nodes.csv'
rel_path = f'testing/graphs/{args.file_sample}_samples-{numeric_N}_numeric-{text_N}_text-relationships.csv'
# print(node_rows[0])
# print(list(node_row.keys()))
# exit()
with open(nodes_path, 'w') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=list(node_row.keys()))
    for row in node_rows:
        writer.writerow(row)
        
with open(rel_path, 'w') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=list(graph_row.keys()))
    for row in rel_rows:
        writer.writerow(row)