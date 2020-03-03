# Extraction Modules:
# from components.parse_files.parser import Parser
# from components.utils.test import SimpleTest

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
# import keras
# import tensorflow as tf
# import keras.backend as K
import argparse
from os import path

# Configure argument parser
parser = argparse.ArgumentParser(description='''
                                Tests sella pipeline on given excel spreadsheet.
                                Outputs Confusion matrix of column similarity test
                                 into testing/confusion_results.
                                Use command line arguments to configure different test types.
                                 ''')
parser.add_argument('-f', '--filename', default='plasmidsDB.xlsx', help='Specify Excel spreadsheet name in data_corpus directory')
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
filename = f'data_corpus/{args.filename}'
if not path.exists(filename):
    print('File not in data_corpus directory!')
assert path.exists(filename)

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
    
    
# Generate Column Summary Vectors
if args.standard and args.numeric:
    SummaryClass = Features
    SUMMARY_TYPE = 'standard_summary'
else:
    SummaryClass = ExtendedSummary
    SUMMARY_TYPE = 'extended_summary'



# TODO: Integrate Clustering Modules.

                