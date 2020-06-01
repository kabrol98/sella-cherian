
import pickle
from components.extended_summaries import extended_summary

# Evaluation Modules
from sklearn.metrics import silhouette_score, make_scorer
from sklearn.model_selection import GridSearchCV

from math import floor, sqrt

from sklearn.cluster import KMeans
from sklearn.cluster import OPTICS

import matplotlib.pyplot as plt
# Extract files from etl

filepath = 'etl/tmp/etl_summaries_5_2020-05-26.pickle'

with open(filepath, "rb") as f:
    data = pickle.load(f)

clusters = KMeans(n_clusters=7).fit(data['numeric']).predict(data['numeric'])

def score_fn(estimator, X):
    clusters = estimator.fit_predict(X)
    score = silhouette_score(X, clusters)
    return score

N = len(data['numeric'])
print(f'{N} clusters')
param_grid = { 'n_clusters': range(floor(sqrt(N)/3), floor(3*sqrt(N)))}
search = GridSearchCV(KMeans(), param_grid, scoring = score_fn,cv=2)
search.fit(data['numeric'])

means = search.cv_results_['mean_test_score']
std = search.cv_results_['std_test_score']
params = search.cv_results_['params']

for m,s,p in zip(means,std,params):
    print(f'{m} +/- {s**2} for {p}')

x = list(map(lambda x: x['n_clusters'], params))
print(len(x), len(means), len(std**2))

plt.errorbar(x,means,yerr=std*2)
plt.title(f'KMeans testing: {N} columns, silhouette score by n_clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score (with error bars)')
plt.savefig('testing/cluster_testing/kmeans_cluster_errors')