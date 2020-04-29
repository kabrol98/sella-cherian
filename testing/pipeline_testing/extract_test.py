import keras
import tensorflow as tf

import keras.backend as K

from components.parse_files.parser import Parser
from components.utils.test import SimpleTest
from components.column_summaries.features import Features
import argparse
import pickle

if __name__ == '__main__':
    with SimpleTest():
        # Argument parse code
        parser = argparse.ArgumentParser(description='Classify and extract data cells and columns.')
        parser.add_argument('-f', '--fname',
                            help='filename to parse')
        parser.add_argument('-i', '--fid', type=int, default=-1,
                            help='file id in given list')
        pargs = parser.parse_args()
        args = vars(pargs)
        print(args)
        # file names.
        files = [
            "data_corpus/00sumdat_copy.xlsx",
            "data_corpus/1-1-10Na-Kinetic Curves.xlsx",
            "data_corpus/Aexperiment.xlsx",
            "data_corpus/CMOP.xlsx",
            "data_corpus/dnhperftbls.xlsx",
            "data_corpus/eScience and Cyberinfrastructure_Same Size.xlsx",
            "data_corpus/Fully Duplicate ReorderedColumns.xlsx",
            "data_corpus/GOA2007_Leg1.xlsx",
            "data_corpus/m0902053;4001_4003.xlsx",
            "data_corpus/metabolite data.xlsx",
            "data_corpus/Michela MayJune RegularCruise Containment.xlsx",
            "data_corpus/ML-W0704A-nutrients1.xlsx",
            "data_corpus/plasmidsDB.xlsx"
        ]
        model_path = "models/NeuralNetwork/vertical_lstm.h5"
        if args['fname'] is not None:
            if args['fname'] not in files:
                print('File not found in data corpus!')
                exit()
            filename = f"data_corpus/{args['fname']}"
        else:
            filename = files[args['fid']]
        print(f'Extracting columns from {filename}...')
        
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        model = keras.models.load_model(model_path)

        parser = Parser(filename, model)
        results = []
        for res in parser.parse():
            results.append(res)
            print(res)
        pickle.dump(results, open('columns.p', "wb"))
        # feature = Features(res, None)
        # print(feature.data_type, feature.value_range, feature.max, feature.min, feature.mean,
        #         feature.std, feature.mask, feature.common_values0, feature.common_frequencies0,
        #         feature.common_values1, feature.common_frequencies1, feature.unique_num, feature.null_num,
        #         feature.column_metadata, feature.file_metadata, "\n")
