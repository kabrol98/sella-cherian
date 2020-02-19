import os

import numpy as np
import pandas as pd
from collections import OrderedDict, defaultdict
import random
import math

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation, concatenate
from keras.optimizers import Adam
from keras.utils import to_categorical, plot_model
from keras import backend as K, Model

import pickle

PAD_CONTENT = 0
PAD_TAG = 0

all_col_names = ["file_name", "is_alpha", "text_in_header", "is_num", "is_alphanum", "is_blank", "is_nullDefault", "all_small", "all_capital", "starts_capital", "contain_colon", "contain_special", "text_length", "year_range", "has_merge_cell", "left_align", "center_align", "right_align", "italics_font", "underline_font", "bold_font", "left_alpha", "left_in_header", "left_num", "left_alphanum", "left_blank", "above_alpha", "above_in_header", "above_num", "above_alphanum", "above_blank", "below_alpha", "below_in_header", "below_num", "below_alphanum", "below_blank", "right_alpha", "right_in_header", "right_num", "right_alphanum", "right_blank", "label"]

data_col_names = ["is_blank", "bold_font","below_blank","has_merge_cell","above_alpha","left_align","right_blank","above_blank","above_num","above_alphanum","right_align","underline_font","below_num","left_alpha","above_in_header","left_num","all_small","is_alpha","right_num","text_in_header","is_num"]

FEATURE_LENGTH = len(data_col_names)

all_cols = {}
all_rows = {}

def get_context(cell_name, df):
    if not all_cols or not all_rows:
        for idx, row in df.iterrows():
            string = row["file_name"]
            column_name = string.split("[")[0] + string.split(",")[1]
            row_name = string.split("[")[0] + string.split(",")[0]
            cell = row[data_col_names].values.tolist()
            if column_name not in all_cols:
                all_cols[column_name] = [cell]
            else:
                all_cols[column_name].append(cell)
            if row_name not in all_rows:
                all_rows[row_name] = [cell]
            else:
                all_rows[row_name].append(cell)

    cell_col_name = cell_name.split("[")[0] + cell_name.split(",")[1]
    cell_row_name = cell_name.split("[")[0] + cell_name.split(",")[0]
    return all_cols[cell_col_name], all_rows[cell_row_name]

all_data = pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), "newTraining.data")), names=all_col_names, header=None, delimiter=r"\s+")

all_data.select_dtypes(exclude=['object', 'datetime']) + 1

all_x = []
all_y = []


for row_idx, row in all_data.iterrows():
    vertical_context, horizontal_context = get_context(row["file_name"], all_data)
    label = row["label"]
    all_x.append([vertical_context, horizontal_context])
    all_y.append(label)

# pickle.dump(all_x, open('all_x.pickle', 'wb'))
# pickle.dump(all_y, open('all_y.pickle', 'wb'))

max_len_vertical = -1
max_len_horizontal = -1
for vertical_context, horizontal_context in all_x:
    max_len_vertical = max(len(vertical_context), max_len_vertical)
    max_len_horizontal = max(len(horizontal_context), max_len_horizontal)
for vertical_context, horizontal_context in all_x:
    while len(vertical_context) < max_len_vertical:
        vertical_context.append([PAD_CONTENT] * FEATURE_LENGTH)
    while len(horizontal_context) < max_len_horizontal:
        horizontal_context.append([PAD_CONTENT] * FEATURE_LENGTH)

all_x = np.array(all_x)

encoder = LabelEncoder()
encoder.fit(all_y)
all_y = encoder.transform(all_y)

x_train, x_test, y_train, y_test = train_test_split(all_x, all_y, test_size=0.2)

CLASS_NUM = 5+1
EPOCHS = 20

vertial_model = load_model("vertical_lstm.h5")
vertial_model.name = 'vertial_model'
horizontal_model = load_model("horizontal_lstm.h5")
horizontal_model.name = 'horizontal_model'

models = [vertial_model, horizontal_model]
for model in [vertial_model, horizontal_model]:
    for layer in model.layers:
        print(layer.name)
        layer.trainable = False
        layer.name = 'ensemble_' + layer.name + '_rand_' + str(random.randint(1,1000))
        print(layer.name)
    print(model.name)

ensemble_visible = [model.input for model in models]
# concatenate merge output from each model
ensemble_outputs = [model.output for model in models]
merge = concatenate(ensemble_outputs)
hidden = Dense(10, activation='relu')(merge)
output = Dense(3, activation='softmax')(hidden)
model = Model(inputs=ensemble_visible, outputs=output)
# plot graph of ensemble
plot_model(model, show_shapes=True, to_file='model_graph.png')
# compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=EPOCHS, verbose=2)
model.save(os.path.abspath(os.path.join(os.path.dirname(__file__), "ensemble.h5")))

scores = model.evaluate(x_test, to_categorical(y_test, CLASS_NUM))
predictions = model.predict(x_test)
print(confusion_matrix(y_test.flatten(), y_pred=np.apply_along_axis(lambda arr: np.argmax(arr), 1, predictions.reshape(-1, predictions.shape[-1]))))
print(f"{model.metrics_names[1]}: {scores[1] * 100}")


# model = Sequential()
# # model.add(InputLayer(input_shape=(max_len, FEATURE_LENGTH)))
# model.add(Bidirectional(LSTM(12, return_sequences=True, dropout=DROPOUT, recurrent_dropout=DROPOUT), input_shape=(None, FEATURE_LENGTH)))
# model.add(TimeDistributed(Dense(CLASS_NUM)))
# model.add(Activation('softmax'))
# model.compile(loss='categorical_crossentropy',
#               optimizer=Adam(),
#               sample_weight_mode="temporal",
#               metrics=['accuracy'])#, ignore_accuracy_of_class(PAD_TAG)])
# model.summary()
# model.fit(x_train, to_categorical(y_train, CLASS_NUM), batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2, verbose=2)
# model.save(os.path.abspath(os.path.join(os.path.dirname(__file__), "horizontal_lstm.h5")))
# scores = model.evaluate(x_test, to_categorical(y_test, CLASS_NUM))
# predictions = model.predict(x_test)
# print(confusion_matrix(y_test.flatten(), y_pred=np.apply_along_axis(lambda arr: np.argmax(arr), 1, predictions.reshape(-1, predictions.shape[-1]))))
# print(f"{model.metrics_names[1]}: {scores[1] * 100}")


