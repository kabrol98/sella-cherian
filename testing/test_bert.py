import pickle
from components.extract_column.column import Column
from components.extended_summaries.bert_summary import BertSummary
from sklearn.metrics.pairwise import cosine_similarity
import codecs, json 
import numpy as np

with open('columns.p', 'rb') as f:
    columns = pickle.load(f)
print("Extracted Columns...")

BertsList = [BertSummary(col) for col in columns]
print("Built BertSummaries...")

embedList = [bert.bert_vector for bert in BertsList]
print("Extracted vectors...")

embedList = np.array(embedList)
cosine_matrix = cosine_similarity(embedList)
print("Computed Cosines!...")
print(cosine_matrix)

file_path = './similarities.json'
json.dump(cosine_matrix.tolist(), codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4) ### this saves the array in .json format
