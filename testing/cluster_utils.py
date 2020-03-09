
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
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS


CLUSTER_PARAMS={
    'DBSCAN':{'eps':4, 'min_samples':2},
    'OPTICS':{'min_samples':2},
    'KMEANS':{'n_clusters':10},
    'EM':{'n_components':10}
}

def test_KMeans(columns, pkey, params, rng):
    N = columns.shape[0]
    cluster_obj= []
    for val in rng:
        params[pkey] = val
        c_obj = MiniBatchKMeans(n_clusters=params['n_clusters'])
        cluster_obj.append(c_obj)
    clusters = [o.fit_predict(columns) for o in cluster_obj]
    return clusters

def test_EM(columns, pkey, params, rng):
    N = columns.shape[0]
    cluster_obj= []
    for val in rng:
        params[pkey] = val
        c_obj = GaussianMixture(n_components=params['n_components'])
        cluster_obj.append(c_obj)
    clusters = [o.fit_predict(columns) for o in cluster_obj]
    return clusters

def test_DBSCAN(columns,pkey, params, rng):
    cluster_obj= []
    for val in rng:
        params[pkey] = val
        c_obj = DBSCAN(eps=params['eps'])
        cluster_obj.append(c_obj)
    clusters = [o.fit_predict(columns) for o in cluster_obj]
    # print(columns.shape, clusters[0].shape)
    return clusters

def test_OPTICS(columns, pkey, params, rng):
    cluster_obj= []
    for val in rng:
        params[pkey] = val
        c_obj = OPTICS(min_samples=params['min_samples'])
        cluster_obj.append(c_obj)
    clusters = [o.fit_predict(columns) for o in cluster_obj]
    # print(columns.shape, clusters[0].shape)
    return clusters

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

def split_on_cluster(matrix, assignments, labels, sheets):
    # TODO:Finish
    K = np.max(assignments) + 1
    # print(assignments)
    cluster_set = np.array([matrix[assignments==i] for i in range(K)])
    label_set = np.array([np.extract([assignments==i], labels) for i in range(K)])
    sheet_set = np.array([np.extract([assignments==i], sheets) for i in range(K)])
    
    return cluster_set, label_set, sheet_set
