import keras
import tensorflow as tf

import keras.backend as K

from components.parse_files.parser import Parser
from components.utils.test import SimpleTest

with SimpleTest():
    files = ["data_corpus/00sumdat_copy.xlsx", "data_corpus/1-1-10Na-Kinetic Curves.xlsx", "data_corpus/Aexperiment.xlsx"]
    model_path = "models/NeuralNetwork/lstm.h5"

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    model = keras.models.load_model(model_path)

    parser = Parser(files[0], model)
    print(parser.parse())

