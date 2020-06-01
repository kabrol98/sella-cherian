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

from etl.extract import extract
from etl.summaries import summaries
# from clustering import clustering
# from similarity import similarity

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
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument('-extraction', dest='stage', action="store_const", const=0)
    stage.add_argument('-summaries', dest='stage', action="store_const", const=1)
    stage.add_argument('-clustering', dest='stage', action="store_const", const=2)
    stage.add_argument('-similarity', dest='stage', action="store_const", const=3)
    parser.set_defaults(stage=0)
    args = parser.parse_args()
    return args

def save_stage(stage,sample_size, data):
    outpath = f"etl/tmp/etl_{stage}_{sample_size}_{date.today().isoformat()}.pickle"
    with open(outpath, "wb") as f:
        pickle.dump(data,f)
        print(f'saved {stage} stage to {outpath}.')
        
args = parse_etl_args()
stage = STAGE_NAMES[args.stage]
sample_size = args.file_sample

# Stage one: extraction
extraction_results = extract(sample_size)
if stage == 'extraction':
    save_stage(stage,sample_size, extraction_results)
    exit()

summary_results = summaries(extraction_results)
if stage == 'summaries':
    save_stage(stage,sample_size, summary_results)
    exit()

print(f'Stage {stage} not yet supported')
exit()
