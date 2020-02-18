import keras
import tensorflow as tf

import keras.backend as K

from components.parse_files.parser import Parser
from components.utils.test import SimpleTest
from components.column_summaries.features import Features

with SimpleTest():
    files = ["data_corpus/00sumdat_copy.xlsx", "data_corpus/1-1-10Na-Kinetic Curves.xlsx", "data_corpus/Aexperiment.xlsx"]
    model_path = "models/NeuralNetwork/lstm.h5"

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    model = keras.models.load_model(model_path)

    parser = Parser(files[0], model)
    for res in parser.parse():
        # print(res)
        feature = Features(res, None)
        # print(feature.data_type, feature.value_range, feature.max, feature.min, feature.mean,
        #         feature.std, feature.mask, feature.common_values0, feature.common_frequencies0,
        #         feature.common_values1, feature.common_frequencies1, feature.unique_num, feature.null_num,
        #         feature.column_metadata, feature.file_metadata, "\n")
