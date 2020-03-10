
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
from sklearn.cluster import Birch


CLUSTER_PARAMS={
    'DBSCAN':{'eps':lambda N: 0.9, 'min_samples':lambda N: 1},
    'OPTICS':{'min_samples':lambda N:2},
    'KMEANS':{'n_clusters':lambda N: int(sqrt(N))},
    'EM':{'n_components':lambda N: int(sqrt(N))}
    # 'BIRCH':{'threshold':lambda N: 0.5,'branching_factor':lambda N: 50,'n_clusters':lambda N: 3}
}

def test_KMeans(columns, pkey, params, rng):
    N = columns.shape[0]
    cluster_obj= []
    for val in rng:
        params[pkey] = lambda N: val
        c_obj = MiniBatchKMeans(n_clusters=params['n_clusters'](N))
        cluster_obj.append(c_obj)
    clusters = [o.fit_predict(columns) for o in cluster_obj]
    return clusters

def test_BIRCH(columns, pkey, params, rng):
    N = columns.shape[0]
    cluster_obj= []
    for val in rng:
        params[pkey] = lambda N: val
        c_obj = Birch(threshold=params['threshold'](N),branching_factor=params['branching_factor'](N),n_clusters=params['n_clusters'](N))
        cluster_obj.append(c_obj)
    clusters = [o.fit_predict(columns) for o in cluster_obj]
    return clusters

def test_EM(columns, pkey, params, rng):
    N = columns.shape[0]
    cluster_obj= []
    for val in rng:
        params[pkey] = lambda N: val
        c_obj = GaussianMixture(n_components=params['n_components'](N))
        cluster_obj.append(c_obj)
    clusters = [o.fit_predict(columns) for o in cluster_obj]
    return clusters

def test_DBSCAN(columns,pkey, params, rng):
    cluster_obj= []
    N = columns.shape[0]
    for val in rng:
        params[pkey] = lambda N: val
        c_obj = DBSCAN(eps=params['eps'](N), min_samples=params['min_samples'](N))
        cluster_obj.append(c_obj)
    clusters = [o.fit_predict(columns) for o in cluster_obj]
    # print(columns.shape, clusters[0].shape)
    return clusters

def test_OPTICS(columns, pkey, params, rng):
    cluster_obj= []
    N = columns.shape[0]
    for val in rng:
        params[pkey] = lambda N: val
        c_obj = OPTICS(min_samples=params['min_samples'](N))
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

def split_on_cluster(matrix, assignments, labels, sheets, filenames, ids):
    # TODO:Finish
    K = np.max(assignments) + 1
    # print(assignments)
    cluster_set = np.array([matrix[assignments==i] for i in range(K)])
    label_set = np.array([np.extract([assignments==i], labels) for i in range(K)])
    sheet_set = np.array([np.extract([assignments==i], sheets) for i in range(K)])
    file_set = np.array([np.extract([assignments==i], filenames) for i in range(K)])
    id_set = np.array([np.extract([assignments==i], ids) for i in range(K)])
    return cluster_set, label_set, sheet_set, file_set, id_set
