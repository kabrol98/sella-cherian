
import matplotlib.pyplot as plt
import numpy as np
import glob
from sklearn.utils.random import sample_without_replacement as sample
from LocalitySensitiveHashing import LocalitySensitiveHashing
from copy import deepcopy as cpy

# Evaluation Modules
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import silhouette_score

# Clustering Modules
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture

def test_KMeans(columns):
    K_PARAMS = {'k':10}
    N = columns.shape[0]
    k_vals = np.linspace(5,int(2*np.sqrt(N)), num=20, dtype=np.int32)
    params = [cpy(K_PARAMS) for _ in range(len(k_vals))]
    for i in range(len(k_vals)):
        params[i]['k'] = k_vals[i]
    cluster_obj = [MiniBatchKMeans(param['k']) for param in params]
    clusters = [o.fit_predict(columns) for o in cluster_obj]
    print(columns.shape, clusters[0].shape)
    db_scores = [davies_bouldin_score(columns, cluster) for cluster in clusters]
    ch_scores = [calinski_harabasz_score(columns, cluster) for cluster in clusters]
    s_scores = [silhouette_score(columns, cluster) for cluster in clusters]
    
    return db_scores, ch_scores, s_scores, k_vals
    # print(clusters)

def test_LSH(columns):
    LSH_PARAMS = {'k': 10, 'r': 50, 'b': 100}
    N = columns.shape[0]
    k_vals = np.linspace(0,int(2*np.sqrt(N)), num=20, dtype=np.int32)
    params = [cpy(LSH_PARAMS) for _ in range(len(k_vals))]
    for i in range(len(k_vals)):
        params[i]['k'] = k_vals[i]
    cluster_obj = [LSHCluster(param['k'], param['r'], param['b']) for param in params]
    clusters = [o.fit_transform(columns) for o in cluster_obj]
    print(clusters)
    exit()
    db_scores = [davies_bouldin_score(columns, cluster) for cluster in clusters]
    ch_scores = [calinski_harabasz_score(columns, cluster) for cluster in clusters]
    s_scores = [silhouette_score(columns, cluster) for cluster in clusters]
    
    return db_scores, ch_scores, s_scores, k_vals

def plot_scores(params, scores, scorename, ax):
    ax.plot(params,scores)
    ax.set_title(scorename)
    return ax


class LSHCluster:
    def __init__(self, k, r, b):
        self.K = k
        self.r = r
        self.b = b
        
    def fit_transform(self, X):
        N = X.shape[0]
        names = np.arange(0,N,1)
        data = np.insert(X,0,names,axis=1)
        # exit()
        # with open('temp.csv') as f:
        np.savetxt('temp.csv', data, delimiter=',')
        lsh = LocalitySensitiveHashing(
                   datafile = 'temp.csv',
                   dim = X.shape[0],
                   r = self.r,
                   b = self.b,
                   expected_num_of_clusters = self.K,
          )
        similarity_groups = lsh.lsh_basic_for_neighborhood_clusters()
        coalesced_similarity_groups = lsh.merge_similarity_groups_with_coalescence( similarity_groups )
        merged_similarity_groups = lsh.merge_similarity_groups_with_l2norm_sample_based( coalesced_similarity_groups )
        return merged_similarity_groups

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
