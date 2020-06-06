import numpy as np
from sklearn.cluster import KMeans
from math import sqrt
def split_on_cluster(matrix,assignments, ids):
    # TODO:Finish
    K = np.max(assignments) + 1
    # print(assignments)
    cluster_set = np.array([matrix[assignments==i] for i in range(K)])
    # label_set = np.array([np.extract([assignments==i], labels) for i in range(K)])
    # sheet_set = np.array([np.extract([assignments==i], sheets) for i in range(K)])
    # file_set = np.array([np.extract([assignments==i], filenames) for i in range(K)])
    id_set = np.array([np.extract([assignments==i], ids) for i in range(K)])
    return cluster_set, id_set

def clustering(data):
    numeric_data = data['numeric']
    numeric_names = data['numeric_names']
    text_data = data['text']
    text_names = data['text_names']
    
    numeric_assignments = KMeans(n_clusters=int(sqrt(len(numeric_data)))).fit_predict(numeric_data)
    text_assignments = KMeans(n_clusters=int(sqrt(len(text_data)))).fit_predict(text_data)
    
    numeric_clusters, numeric_ids = split_on_cluster(numeric_data, numeric_assignments, numeric_names)
    text_clusters, text_ids = split_on_cluster(text_data, text_assignments, text_names)
    # print(numeric_clusters)
    return {'numeric_clusters': numeric_clusters,
             'text_clusters': text_clusters
             }
