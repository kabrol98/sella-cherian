import os

import numpy as np
import pandas as pd
from collections import OrderedDict, defaultdict
import random
import math

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras import backend as K

PAD_CONTENT = 0
PAD_TAG = 0

all_col_names = ["file_name", "is_alpha", "text_in_header", "is_num", "is_alphanum", "is_blank", "is_nullDefault", "all_small", "all_capital", "starts_capital", "contain_colon", "contain_special", "text_length", "year_range", "has_merge_cell", "left_align", "center_align", "right_align", "italics_font", "underline_font", "bold_font", "left_alpha", "left_in_header", "left_num", "left_alphanum", "left_blank", "above_alpha", "above_in_header", "above_num", "above_alphanum", "above_blank", "below_alpha", "below_in_header", "below_num", "below_alphanum", "below_blank", "right_alpha", "right_in_header", "right_num", "right_alphanum", "right_blank", "label"]

data_col_names = ["is_blank", "bold_font","below_blank","has_merge_cell","above_alpha","left_align","right_blank","above_blank","above_num","above_alphanum","right_align","underline_font","below_num","left_alpha","above_in_header","left_num","all_small","is_alpha","right_num","text_in_header","is_num"]

FEATURE_LENGTH = len(data_col_names)



def get_lists(df, row_idxs, is_label = False):
    raw_df = df.iloc[row_idxs]
    if is_label:
        return raw_df[['label']].values.flatten().tolist()
    else:
        return raw_df[data_col_names].values.tolist()

def ignore_accuracy_of_class(class_to_ignore=0):
    """https://stackoverflow.com/questions/47270722/how-to-define-a-custom-accuracy-in-keras-to-ignore-samples-with-a-particular-gol"""
    def ignore_acc(y_true, y_pred):
        y_true_class = K.argmax(y_true)
        y_pred_class = K.argmax(y_pred)

        ignore_mask = K.cast(K.not_equal(y_pred_class, class_to_ignore), 'int32')
        matches = K.cast(K.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
        accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
        return accuracy
    return ignore_acc

def compute_weights(counts_dict):
    """https://datascience.stackexchange.com/questions/13490/how-to-set-class-weights-for-imbalanced-classes-in-keras"""
    total = np.sum(list(counts_dict.values()))
    class_weight = dict()
    for key, val in counts_dict.items():
        if key == 0:
            score = 0
        else:
            score = math.log(total/float(val))
        class_weight[key] = score * 5 if score > 1.0 else 1.0
    return class_weight

all_data = pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), "newTraining.data")), names=all_col_names, header=None, delimiter=r"\s+")

all_data.select_dtypes(exclude=['object', 'datetime']) + 1


rows_set = OrderedDict()
for idx, row in all_data.iterrows():
    string = row["file_name"]
    row_name = string.split("[")[0] + string.split(",")[0]
    if row_name in rows_set:
        rows_set[row_name].append(idx)
    else:
        rows_set[row_name] = [idx]

row_names = rows_set.keys()
# train_column_names = column_names
train_row_names = random.sample(row_names, math.floor(len(row_names) * 0.8))
test_row_names = [name for name in row_names if name not in train_row_names]


x_train = [get_lists(all_data, rows_set[name]) for name in train_row_names]
x_test = [get_lists(all_data, rows_set[name]) for name in test_row_names]
max_len = 0
for l in x_train:
    if len(l) > max_len:
        max_len = len(l)
for l in x_test:
    if len(l) > max_len:
        max_len = len(l)
for l in x_train:
    while len(l) < max_len:
        l.append([PAD_CONTENT] * FEATURE_LENGTH)
for l in x_test:
    while len(l) < max_len:
        l.append([PAD_CONTENT] * FEATURE_LENGTH)
x_train = np.array(x_train)
x_test = np.array(x_test)

encoder = LabelEncoder()
y_train_temp = []
encoder_learner = []
for name in train_row_names:
    res = get_lists(all_data, rows_set[name], True)
    y_train_temp.append(res)
    encoder_learner.extend(res)
encoder.fit(encoder_learner)
# ['CH' 'DC' 'DE' 'DS' 'NDC']
y_train = []
for y in y_train_temp:
    res = (encoder.transform(y)+1).tolist()
    while len(res) < max_len:
        res.append(PAD_TAG)
    y_train.append(res)
label_weights = defaultdict(int)
for j in y_train:
    for k in j:
        label_weights[k] += 1
label_weights = compute_weights(label_weights)
y_train = np.array(y_train)

def vec_func(x):
    # print(x, label_weights[x])
    return label_weights[x]
vector_func = np.vectorize(vec_func)
sample_weights = vector_func(y_train)

# print(y_train)
# exit(1)
y_test = []
for name in test_row_names:
    res = (encoder.transform(get_lists(all_data, rows_set[name], True)) + 1).tolist()
    while len(res) < max_len:
        res.append(PAD_TAG)
    y_test.append(res)
y_test = np.array(y_test)

CLASS_NUM = 5 + 1
BATCH_SIZE = 10
EPOCHS = 20
DROPOUT = 0.1

model = Sequential()
# model.add(InputLayer(input_shape=(max_len, FEATURE_LENGTH)))
model.add(Bidirectional(LSTM(12, return_sequences=True, dropout=DROPOUT, recurrent_dropout=DROPOUT), input_shape=(None, FEATURE_LENGTH), name="horizontal_bidirectional"))
model.add(Dense(CLASS_NUM, name="horizontal_dense"))
model.add(Activation('softmax', name="horizontal_activation"))
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              sample_weight_mode="temporal",
              metrics=['accuracy'])#, ignore_accuracy_of_class(PAD_TAG)])
model.summary()
model.fit(x_train, to_categorical(y_train, CLASS_NUM), batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2, verbose=2, sample_weight=sample_weights)
model.save(os.path.abspath(os.path.join(os.path.dirname(__file__), "horizontal_lstm.h5")))
scores = model.evaluate(x_test, to_categorical(y_test, CLASS_NUM))
predictions = model.predict(x_test)
print(confusion_matrix(y_test.flatten(), y_pred=np.apply_along_axis(lambda arr: np.argmax(arr), 1, predictions.reshape(-1, predictions.shape[-1]))))
print(f"{model.metrics_names[1]}: {scores[1] * 100}")


