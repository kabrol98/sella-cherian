DESCRIPTION = """This script should allow you to transform
a subset of spreadsheets from the data corpus 
into an intermediate result from any stage in sella's 
pipeline. It can also be used to familiarize yourself 
with any given component of the Sella system.

NOTE: for any stage apart from extraction, bert-serving must be running on your machine.
"""
# # Clustering Modules
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import MiniBatchKMeans
# from sklearn.mixture import GaussianMixture
# from sklearn.cluster import DBSCAN
import numpy as np
from etl.extract import extract, DESCRIPTION as desc0
from etl.summaries import summaries, DESCRIPTION as desc1
from etl.clustering import clustering, DESCRIPTION as desc2
# from clustering import clustering
from etl.similarity import similarity
import csv
# Python Modules
import argparse
# from os import path
# from math import sqrt
import pickle
from datetime import date
# from glob import glob
# from matplotlib import pyplot as plt
# import collections

STAGE_NAMES = ['extraction','summaries','clustering','similarity']

def parse_etl_args():
    # Configure argument parser
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('-S', '--file_sample', default=5, type=float, help="Pick number of files to randomly sample")
    parser.add_argument('-source', '-dir', dest="directory", default=None)
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument('-extraction', dest='stage', action="store_const", const=0, help=desc0)
    stage.add_argument('-summaries', dest='stage', action="store_const", const=1, help=desc1)
    stage.add_argument('-clustering', dest='stage', action="store_const", const=2, help=desc2)
    stage.add_argument('-similarity', dest='stage', action="store_const", const=3)
    
    parser.add_argument('-load', dest='loadFile', type=int)
    parser.set_defaults(stage=0, start=0)
    args = parser.parse_args()
    return args

def save_stage(stage,sample_size, data):
    # data['stage'] = stage
    outpath = f"etl/tmp/etl_{stage}_{sample_size}_{date.today().isoformat()}.pickle"
    with open(outpath, "wb") as f:
        pickle.dump(data,f)
        print(f'saved {stage} stage to {outpath}.')
    exit()
        
args = parse_etl_args()
stage = STAGE_NAMES[args.stage]
sample_size = args.file_sample
dir = args.directory

# Stage one: extraction
extraction_results = extract(sample_size, dir)
if stage == 'extraction':
    save_stage(stage,sample_size, extraction_results)
# Stage two: summaries
summary_results = summaries(extraction_results)
if stage == 'summaries':
    save_stage(stage,sample_size, summary_results)
# Stage three: clustering
clustering_results = clustering(summary_results)
if stage == 'clustering':
    save_stage(stage,sample_size, clustering_results)
# Stage four: similarity
similarity_results = similarity(clustering_results)
outpath = f"etl/tmp/etl_column_graph_{sample_size}_{date.today().isoformat()}.csv"
arr = np.asarray(similarity_results['columns'])
print(arr)
np.savetxt(outpath, arr ,fmt='%s', delimiter=",")
outpath = f"etl/tmp/etl_sheet_graph_{sample_size}_{date.today().isoformat()}.csv"
arr = np.asarray(similarity_results['sheets'])
print(arr)
np.savetxt(outpath, arr ,fmt='%s', delimiter=",")

# with open(outpath, 'wb') as f:
#     csv.writer(f).writerows(similarity_results)

exit()

print(f'Stage {stage} not yet supported')
