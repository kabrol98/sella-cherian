DESCRIPTION="""Extraction stage
Stage one will take N file samples from the data_corpus directory,
classify each cell in each spreadsheet, and extract column objects
from the spreadsheets. Column is a custom class defined in 
components/extract_column/column.py. It primarily will contain 
file metadata along with a list of cell objects (also a custom class).
"""

from components.parse_files.parser import Parser
from components.utils.test import SimpleTest

from os import path
import keras
import tensorflow as tf
import keras.backend as K

from testing.test_utils.testing_utils import sample_dataset

def extract(N, dir):
    if dir is not None:
        # TODO: handle this.
        print('unhandled')
    # Test for valid filenames
    filenames = sample_dataset(N, None)
    NUM_FILES=len(filenames)
    for n in filenames:
        if not path.exists(n):
            print(f'File {n} does not exist!')
        assert path.exists(n)

    model_path = "models/NeuralNetwork/vertical_lstm.h5"
    print(f'Extracting columns from dataset...')

    results = []
    for filename in filenames:
        with SimpleTest():

            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
            model = keras.models.load_model(model_path)

            parser = Parser(filename, model)
            for res in parser.parse():
                results.append(res)
            print(f'Extracted {len(results)} columns from {filename}...')
    return results