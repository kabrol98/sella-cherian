import pickle
from components.extract_column.column import Column
# from components.bert_summaries.bert_summary import BertSummary
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from components.cell_labeling.cell_compact import ContentType
from components.numerical_summaries.any_summary import AnySummary
import codecs, json 
import numpy as np


def compute_sim_results(embedList, path=None):
    cosine_matrix = cosine_similarity(embedList)
    print("Computed Cosines...")
    print(cosine_matrix)

    if path != None:
        json.dump(cosine_matrix.tolist(), codecs.open(path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4) ### this saves the array in .json format
    return cosine_matrix
with open('columns.p', 'rb') as f:
    columns = pickle.load(f)
print("Extracted Columns...")

numericals = np.extract([x.type==ContentType.NUMERIC for x in columns], columns)
berts = np.extract([x.type!=ContentType.NUMERIC for x in columns], columns)
# print(columns)
print(f'Found {len(numericals)} numerical columns, {len(berts)} text columns')
numericalSummaries = [AnySummary(c) for c in numericals]
# bertSummaries = [AnySummary(c) for c in berts]
print('Generated summaries...')
numericalVectors = np.array([c.vectorize() for c in numericalSummaries])
# bertVectors = np.array([c.vectorize() for c in numericalSummaries])
print(numericalVectors.shape)
cosine_matrix = compute_sim_results(numericalVectors)
# compute_sim_results(bertVectors)

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
ax.set_title('')
ax.imshow(cosine_matrix, cmap='Pastel1')
fig.tight_layout()
# plt.show()
plt.savefig('rel-numeric-noclusters.png')
