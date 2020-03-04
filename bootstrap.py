import random
from typing import List
import pickle

import keras
import tensorflow as tf

from components.cell_labeling.cell_compact import ContentType
from components.extract_column.column import Column
from components.parse_files.parser import Parser


def create_column_dicts(cols: List[Column]):
    def get_sublist(indices, l):
        return [l[i] for i in indices]

    def create_hybird_column_dict(col1: Column, col2: Column):
        col2_indices = [i for i in range(len(col2.content_cells))]
        col1_indices = [i for i in range(len(col1.content_cells))]
        col2_rand_index = random.sample(col2_indices, 1)[0]
        col2_part1_indices = col2_indices[:col2_rand_index]
        # col2_part2_indices = col2_indices[col2_rand_index:]
        col1_rand_index = random.sample(col1_indices, 1)[0]
        col1_part1_indices = col1_indices[col1_rand_index:]
        # col1_part2_indices = col1_indices[col1_rand_index:]
        part1_content_cells = get_sublist(col2_part1_indices, col2.content_cells)
        part1_content_cells.extend(get_sublist(col1_part1_indices, col1.content_cells))
        # part2_content_cells = get_sublist(col2.content_cells, col2_part2_indices) \
        #     .extend(get_sublist(col1.content_cells, col1_part2_indices))
        if len(part1_content_cells) == 0:
            return None, None
        col_mixed = Column()
        col_mixed.type = col2.type
        col_mixed.header_cells = col1.header_cells
        col_mixed.content_cells = part1_content_cells
        col_mixed.starting_cell = part1_content_cells[0]
        col_mixed.ending_cell = part1_content_cells[-1]
        d1 = {
            'original_col': col2,
            'mixed_col': col_mixed,
            'percentage': len(col2_part1_indices) / len(part1_content_cells)
        }
        d2 = {
            'original_col': col1,
            'mixed_col': col_mixed,
            'percentage': len(col1_part1_indices) / len(part1_content_cells)
        }
        return d1, d2

    all_indices = {i for i in range(len(cols))}
    columns = []

    for idx, col in enumerate(cols):
        all_indices.remove(idx)
        rand_idx = random.choice(list(all_indices))
        selected_col = numeric_columns[rand_idx]
        t1, t2 = create_hybird_column_dict(col, selected_col)
        if t1 is not None:
            columns.append(t1)
        if t2 is not None:
            columns.append(t2)
        all_indices.add(idx)
    return columns


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
    "data_corpus/plasmidsDB.xlsx",
    "data_corpus/PollengraincountsJC_copy containment.xlsx",
    "data_corpus/serine categorized.xlsx",
    "data_corpus/TD220.xlsx",
    # "data_corpus/Test 4.xlsx"
]
model_path = "models/NeuralNetwork/vertical_lstm.h5"

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
model = keras.models.load_model(model_path)

numeric_columns = []
string_columns = []

for file in files:
    parser = Parser(file, model)
    for col in parser.parse():
        if col.type == ContentType.NUMERIC:
            numeric_columns.append(col)
        elif col.type == ContentType.STRING:
            string_columns.append(col)
        else:
            print('Found empty col')
            pass
            # raise ValueError('Null col should not exist')

pickle.dump(create_column_dicts(numeric_columns), open('numeric_bootstrapped_columns.p', 'wb'))
pickle.dump(create_column_dicts(string_columns), open('string_bootstrapped_columns.p', 'wb'))
