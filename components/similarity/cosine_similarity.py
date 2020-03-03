import os, sys

from components.extract_column.column import Column
import pickle
# from components.extended_summaries.bert_summary import BertSummary
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from components.cell_labeling.cell_compact import ContentType
from components.extended_summaries.extended_summary import ExtendedSummary
import codecs, json 
import numpy as np


class CosineSimilarity:
    def __init__(self, 
                 columns: np.array,
                 file_name: np.array,
                 col_names: [],
                 summary_type: str,
                 data_type: str,
                 verbose=False):
        self.data = columns
        self.filename = file_ame,
        self.colnames = col_names,
        self.summarytype = summary_type
        self.datatype = data_type
        self.verbose = verbose
        self.cosine_mat = None
        self.compute_sim()
    
    def compute_sim(self):
        cosine_matrix = cosine_similarity(self.data)
        if self.verbose:
            print("Computed Cosines...")
        self.cosine_mat = cosine_matrix
    
    def save_results(self, pathname):
        labels = self.colnames
        cosine_matrix = self.cosine_mat
        
        fig, ax = plt.subplots()
        # ax.imshow(cosine_matrix)
        labels = [c.serialize() for c in numericalSummaries]
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        # ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = ax.text(j, i, "{0:.1f}".format(cosine_matrix[i, j]),
                            ha="center", va="center", color="b")
        plot_title = f'{self.filename}x{self.summarytype}x{self.datatype}'
        ax.set_title(plot_title)
        ax.imshow(cosine_matrix, cmap='Pastel1')
        fig.tight_layout()
        plt.savefig(f'testing/{pathname}')
        print(f'Saved figure {plot_title} to {pathname}')

# def compute_sim_results(embedList, path=None):
#     cosine_matrix = cosine_similarity(embedList)
#     print("Computed Cosines...")
#     print(cosine_matrix)

#     if path != None:
#         json.dump(cosine_matrix.tolist(), codecs.open(path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4) ### this saves the array in .json format
#     return cosine_matrix


# with open('columns.p', 'rb') as f:
#     columns = pickle.load(f)
# print("Extracted Columns...")

# numericals = np.extract([x.type==ContentType.NUMERIC for x in columns], columns)
# berts = np.extract([x.type!=ContentType.NUMERIC for x in columns], columns)
# # print(columns)
# print(f'Found {len(numericals)} numerical columns, {len(berts)} text columns')
# numericalSummaries = [ExtendedSummary(c) for c in numericals]
# # bertSummaries = [ExtendedSummary(c) for c in berts]
# print('Generated summaries...')
# numericalVectors = np.array([c.vectorize() for c in numericalSummaries])
# # bertVectors = np.array([c.vectorize() for c in numericalSummaries])
# print(numericalVectors.shape)
# cosine_matrix = compute_sim_results(numericalVectors)
# # compute_sim_results(bertVectors)

# fig, ax = plt.subplots()
# # ax.imshow(cosine_matrix)
# labels = [c.serialize() for c in numericalSummaries]
# ax.set_xticks(np.arange(len(labels)))
# ax.set_yticks(np.arange(len(labels)))
# # ax.set_xticklabels(labels)
# ax.set_yticklabels(labels)
# for i in range(len(labels)):
#     for j in range(len(labels)):
#         text = ax.text(j, i, "{0:.1f}".format(cosine_matrix[i, j]),
#                        ha="center", va="center", color="b")
# ax.set_title('Numeric Columns/ Entended Summaries/ No Clustering')
# ax.imshow(cosine_matrix, cmap='Pastel1')
# fig.tight_layout()
# # plt.show()
# plt.savefig('testing/rel-numeric-noclusters.png')
