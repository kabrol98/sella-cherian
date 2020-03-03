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
                 columns: np.array,
                 verbose=False):
        self.data = columns
        self.verbose = verbose
        self.cosine_matrix = None
        self.compute_sim()
    
    def compute_sim(self):
        cosine_matrix = cosine_similarity(self.data)
        if self.verbose:
            print("Computed Cosines...")
        self.cosine_matrix = cosine_matrix
    
