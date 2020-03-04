import os, sys

from components.extract_column.column import Column
import pickle
# from components.extended_summaries.bert_summary import BertSummary
from sklearn.metrics.pairwise import cosine_similarity
from components.cell_labeling.cell_compact import ContentType
from components.extended_summaries.extended_summary import ExtendedSummary
import codecs, json 
import numpy as np


class CosineSimilarity:
    def __init__(self, 
                 cluster_set: np.array,
                 verbose=False):
        self.data = cluster_set
        self.verbose = verbose
        self.cosine_set = None
        self.compute_sim()
    
    def compute_sim(self):
        K = self.data.shape[0]
        if self.verbose:
            print(f'Attempting to compute similarities for each i in {self.data[0]}, {self.data[1]}')
        cosine_set = np.array([cosine_similarity(self.data[i]) for i in range(K)])
        if self.verbose:
            print("Computed Cosines...")
        self.cosine_set = cosine_set
    
