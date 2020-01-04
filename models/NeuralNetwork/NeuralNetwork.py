import numpy
import pandas
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import pandas as pd
import numpy as np
from keras.models import load_model
from keras.layers import Dense, Activation

colnames = ["file_name","is_alpha", "text_in_header", "is_num","is_alphanum","is_blank", "is_nullDefault", "all_small","all_capital","starts_capital", "contain_colon", "contain_special","text_length","year_range", "has_merge_cell", "right_align","left_align","center_align", "italics_font", "underline_font","bold_font","left_alpha", "left_in_header", "left_num","left_alphanum","left_blank", "above_alpha", "above_in_header","above_num","above_alphanum", "above_blank", "below_alpha","below_in_header","below_num", "below_alphanum", "below_blank","right_alpha","right_in_header", "right_num", "right_alphanum","right_blank","label"]

train = pandas.read_csv("train0.7.data", names=colnames, header=None, delimiter=r"\s+")
test = pandas.read_csv("test0.3.data", names=colnames, header=None, delimiter=r"\s+")

train = pandas.DataFrame(train)
test = pandas.DataFrame(test)

trainfeature = train[["is_alpha", "text_in_header", "is_num","is_alphanum","is_blank", "is_nullDefault", "all_small","all_capital","starts_capital", "contain_colon", "contain_special","text_length","year_range", "has_merge_cell", "right_align","left_align","center_align", "italics_font", "underline_font","bold_font","left_alpha", "left_in_header", "left_num","left_alphanum","left_blank", "above_alpha", "above_in_header","above_num","above_alphanum", "above_blank", "below_alpha","below_in_header","below_num", "below_alphanum", "below_blank","right_alpha","right_in_header", "right_num", "right_alphanum","right_blank"]]
X_train = trainfeature[["is_blank", "bold_font","below_blank","has_merge_cell","above_alpha","left_align","right_blank","above_blank","above_num","above_alphanum","right_align","underline_font","below_num","left_alpha","above_in_header","left_num","all_small","is_alpha","right_num","text_in_header","is_num"]]
X_train = X_train.values
y_train = train[["label"]]
y_train = numpy.ravel(y_train)
encoder = LabelEncoder()
encoder.fit(y_train)
encoded_Y = encoder.transform(y_train)
y_train = pd.get_dummies(y_train)
y_train = y_train.values

testfeature = test[["is_alpha", "text_in_header", "is_num","is_alphanum","is_blank", "is_nullDefault", "all_small","all_capital","starts_capital", "contain_colon", "contain_special","text_length","year_range", "has_merge_cell", "right_align","left_align","center_align", "italics_font", "underline_font","bold_font","left_alpha", "left_in_header", "left_num","left_alphanum","left_blank", "above_alpha", "above_in_header","above_num","above_alphanum", "above_blank", "below_alpha","below_in_header","below_num", "below_alphanum", "below_blank","right_alpha","right_in_header", "right_num", "right_alphanum","right_blank"]]
X_test = testfeature[["is_blank", "bold_font","below_blank","has_merge_cell","above_alpha","left_align","right_blank","above_blank","above_num","above_alphanum","right_align","underline_font","below_num","left_alpha","above_in_header","left_num","all_small","is_alpha","right_num","text_in_header","is_num"]]
X_test = X_test.values
y_test = test[["label"]]
y_test = numpy.ravel(y_test)
encoder = LabelEncoder()
encoder.fit(y_test)
encoded_Y = encoder.transform(y_test)
y_test = pd.get_dummies(y_test)
y_test = y_test.values

seed = 1234
np.random.seed(seed)
tf.set_random_seed(seed)
sess = tf.Session()
nb_epoch = 500
nb_classes = 5
batch_size = 10

model = Sequential()
model.add(Dense(32, input_dim=21))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
print(y_train)
model = load_model('model0.7.h5')
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#order of label: [CH, DC, DE, DS, NDC]