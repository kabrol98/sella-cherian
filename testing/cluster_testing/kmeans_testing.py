
import pickle
from components.extended_summaries import extended_summary

# Evaluation Modules
from sklearn.metrics import silhouette_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample
from math import floor, sqrt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import OPTICS

import matplotlib.pyplot as plt
# Extract files from etl

filepath = 'etl/tmp/etl_summaries_15.0_2020-06-03.pickle'

with open(filepath, "rb") as f:
    data = pickle.load(f)

def score_fn(estimator, X):
    clusters = estimator.fit_predict(X)
    score = silhouette_score(X, clusters)
    return score


def get_best_score(dataset,n):
    print(f'{n} clusters')
    param_grid = { 'n_clusters': range(floor(sqrt(n)/3), floor(3*sqrt(n)))}
    search = GridSearchCV(KMeans(), param_grid, scoring = score_fn,cv=2)
    search.fit(dataset)

    means = search.cv_results_['mean_test_score']
    std = search.cv_results_['std_test_score']
    params = search.cv_results_['params']
    best = np.argmax(means)
    print(f'best score with {params[best]} is {means[best]}')
    
    return (params[best], means[best])


N = len(data['numeric'])
nums = np.linspace(N/3, N, 30, dtype=int)
print(nums)
params = []
scores = []
for i in nums:
    dataset = resample(data['numeric'], n_samples=i)
    n = i
    x,y = get_best_score(dataset, n)
    params.append(x['n_clusters'])
    scores.append(y)

print(params, nums, scores)
# exit()
    
plt.plot(nums,params)
plt.title(f'KMeans testing: {n} columns, best argument by number of points')
for x,y, s in zip(nums, params, scores):
    l = '{:.2f}'.format(s)
    plt.annotate(l, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
plt.xlabel('Number of Datapoints')
plt.ylabel('Best n_clusters')
plt.savefig('testing/cluster_testing/kmeans_cluster_best')