
import pickle
from components.extended_summaries import extended_summary

# Evaluation Modules
from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score, make_scorer
from sklearn.model_selection import GridSearchCV

from math import floor, sqrt
import numpy as np
import matplotlib.pyplot as plt
# Extract files from etl

filepath = 'etl/tmp/etl_summaries_5_2020-05-26.pickle'

with open(filepath, "rb") as f:
    data = pickle.load(f)

# clusters = GaussianMixture(n_clusters=7).fit(data['numeric']).predict(data['numeric'])

def score_fn(estimator, X):
    clusters = estimator.fit_predict(X)
    score = silhouette_score(X, clusters)
    return score

N = len(data['numeric'])
print(f'{N} columns.')
param_grid = { 'min_samples': range(1,4)}
search = GridSearchCV(OPTICS(), param_grid, scoring = score_fn,cv=2)
search.fit(data['numeric'])

means = search.cv_results_['mean_test_score']
std = search.cv_results_['std_test_score']
params = search.cv_results_['params']

for m,s,p in zip(means,std,params):
    print(f'{m} +/- {s**2} for {p}')

x = list(map(lambda x: x['min_samples'], params))
print(len(x), len(means), len(std**2))

plt.errorbar(x,means,yerr=std*2)
plt.title(f'OPTICS testing: {N} columns, silhouette score by min_samples')
plt.xlabel('minimum ratio of samples')
plt.ylabel('Silhouette Score (with error bars)')
plt.savefig('testing/cluster_testing/gmm_cluster_errors')