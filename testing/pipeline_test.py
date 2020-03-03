# Extraction Modules:
from components.parse_files.parser import Parser
from components.utils.test import SimpleTest

# Summaries Modules:
from components.column_summaries.features import Features
from components.extended_summaries.extended_summary import ExtendedSummary
from components.cell_labeling.cell_compact import ContentType

# Clustering Modules
# TODO: Write cluster modules for testing purposes.

# Silimarity Modules
from components.similarity.cosine_similarity import CosineSimilarity
# TODO: Add similarity modules for testing purposes.

# Python Modules
from enum import Enum
import numpy as np
import keras
import tensorflow as tf
import keras.backend as K
import argparse
from os import path
import matplotlib.pyplot as plt


def plot_results(matrix: np.array,
                 col_names: [],
                 file_name: str,
                 summary_type: str,
                 data_type: str,
                 path_name: str):
    # print(f'received save-path {path_name}')
    fig, ax = plt.subplots()
    # Label axes
    n = len(col_names)
    rng = np.arange(n)
    ax.set_xticks(rng)
    ax.set_yticks(rng)
    ax.set_yticklabels(col_names)
    # Annotate Similarity Values.
    for i in range(n):
        for j in range(n):
            text = ax.text(j, i, "{0:.1f}".format(matrix[i, j]),
                        ha="center", va="center", color="b")
    plot_title = f'{file_name}||{summary_type}||{data_type}'
    ax.set_title(plot_title)
    ax.imshow(matrix, cmap='Pastel1')
    fig.tight_layout()
    plt.savefig(f'testing/confusion_results/{path_name}.png')
    print(f'Saved figure {plot_title} to {path_name}')


# Configure argument parser
parser = argparse.ArgumentParser(description='''
                                Tests sella pipeline on given excel spreadsheet.
                                Outputs Confusion matrix of column similarity test
                                 into testing/confusion_results.
                                Use command line arguments to configure different test types.
                                 ''')
parser.add_argument('-f', '--filename', default='plasmidsDB', help='Specify Excel spreadsheet name in data_corpus directory (Omit .xlsx)')
# Configure summary type
summary_group = parser.add_mutually_exclusive_group(required=True)
summary_group.add_argument('-s', '--standard', action='store_true', help='Use standard column summaries.')
summary_group.add_argument('-e', '--extended', action='store_true',help='Use extended column summaries.')
# configure datatype type
data_group = parser.add_mutually_exclusive_group(required=True)
data_group.add_argument('-n', '--numeric', action='store_true',help='Run tests on numeric columns')
data_group.add_argument('-t', '--text', action='store_true',help='Run tests on text columns')

args = parser.parse_args()
print(args)
# Run Column Extraction.

# Test for valid filename
filename = f'data_corpus/{args.filename}.xlsx'
if not path.exists(filename):
    print(f'File {filename} does not exist!')
assert path.exists(filename)
# # Test for valid summary
# if args.standard and args.text:
#     print('Standard Column Summary does not support text extraction!')
# assert not (args.standard and args.text)


model_path = "models/NeuralNetwork/vertical_lstm.h5"
print(f'Extracting columns from {filename}...')

with SimpleTest():

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    model = keras.models.load_model(model_path)

    parser = Parser(filename, model)
    results = []
    for res in parser.parse():
        results.append(res)
    print(f'Extracted {len(results)} columns!')

# Filter columns by data type.
if args.numeric:
    DATA_TYPE = 'numeric_data'
    type_class = ContentType.NUMERIC
else:
    DATA_TYPE = 'text_data'
    type_class = ContentType.STRING

columns_filtered = np.extract([x.type==type_class for x in results], results)
# Generate Column Summary Vectors.
if args.standard:
    SummaryClass = Features
    SUMMARY_TYPE = 'standard_summary'
else:
    SummaryClass = ExtendedSummary
    SUMMARY_TYPE = 'extended_summary'

columns_summary = [
    SummaryClass(c) for c in columns_filtered
]
columns_vectorized = np.array([c.vector for c in columns_summary])
column_names = [c.header for c in columns_summary]
print(columns_vectorized[0])
# TODO: Integrate Clustering Modules.


# Run Similarity Analysis
# TODO: Implement alternate similarity modules.
SimilarityClass = CosineSimilarity

cosine_matrix = SimilarityClass(columns_vectorized).cosine_matrix


save_path = f'{args.filename}-{SUMMARY_TYPE}-{DATA_TYPE}.png'

# Plot results.
plot_results(
    cosine_matrix,
    column_names,
    args.filename,
    SUMMARY_TYPE,
    DATA_TYPE,
    save_path
)