import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
# Dependencies
import tensorflow as tf
import pandas as pd
import numpy as np
from keras.models import load_model

# Creates a HDF5 file 'my_model.h5'



from keras.layers import Dense, Activation, Dropout

colnames = ["file_name","is_alpha", "text_in_header", "is_num","is_alphanum","is_blank", "is_nullDefault", "all_small","all_capital","starts_capital", "contain_colon", "contain_special","text_length","year_range", "has_merge_cell", "right_align","left_align","center_align", "italics_font", "underline_font","bold_font","left_alpha", "left_in_header", "left_num","left_alphanum","left_blank", "above_alpha", "above_in_header","above_num","above_alphanum", "above_blank", "below_alpha","below_in_header","below_num", "below_alphanum", "below_blank","right_alpha","right_in_header", "right_num", "right_alphanum","right_blank","label"]

d = {1: "1", 2: "2", 3: "3"}

train = pandas.read_csv("train0.7.data", names=colnames, header=None, delimiter=r"\s+")
test = pandas.read_csv("test0.3.data", names=colnames, header=None, delimiter=r"\s+")

#print(data1.dtypes)
#data = pandas.read_csv("newTraining_V2.data", names=colnames, header=None, delimiter=r"\s+")
#data = np.vstack((data1, data2))
# print("data shape", data1.shape)

#features = pandas.DataFrame.loc[:, "1":"40"]

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
#print("encoder", encoded_Y)
# convert integers to dummy variables (i.e. one hot encoded)
y_train = pd.get_dummies(y_train)
y_train = y_train.values

with open('train_data','w') as f:
    for item in X_train:
        f.write(str(item) + "\n")


testfeature = test[["is_alpha", "text_in_header", "is_num","is_alphanum","is_blank", "is_nullDefault", "all_small","all_capital","starts_capital", "contain_colon", "contain_special","text_length","year_range", "has_merge_cell", "right_align","left_align","center_align", "italics_font", "underline_font","bold_font","left_alpha", "left_in_header", "left_num","left_alphanum","left_blank", "above_alpha", "above_in_header","above_num","above_alphanum", "above_blank", "below_alpha","below_in_header","below_num", "below_alphanum", "below_blank","right_alpha","right_in_header", "right_num", "right_alphanum","right_blank"]]
X_test = testfeature[["is_blank", "bold_font","below_blank","has_merge_cell","above_alpha","left_align","right_blank","above_blank","above_num","above_alphanum","right_align","underline_font","below_num","left_alpha","above_in_header","left_num","all_small","is_alpha","right_num","text_in_header","is_num"]]
X_test = X_test.values
y_test = test[["label"]]
y_test = numpy.ravel(y_test)
encoder = LabelEncoder()
encoder.fit(y_test)
encoded_Y = encoder.transform(y_test)
print("encoder", encoded_Y)
# convert integers to dummy variables (i.e. one hot encoded)
y_test = pd.get_dummies(y_test)
y_test = y_test.values

# print("feature 21 tyle as matrix", type(feature21))


seed = 1234
np.random.seed(seed)
tf.set_random_seed(seed)

# seed = 7
# numpy.random.seed(seed)
# load dataset

#print(label)


print(y_train)
print(X_train.shape)
print(y_train.shape)
sess = tf.Session()

nb_epoch = 500
nb_classes = 5
batch_size = 10

model = Sequential()
model.add(Dense(32, input_dim=21))
model.add(Activation('relu'))
# model.add(Dropout(.2))

model.add(Dense(64))
model.add(Activation('relu'))
# model.add(Dropout(.5))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, epochs=nb_epoch, verbose=2)
model.save('model0.7.h5')
score = model.evaluate(X_test, y_test)
print('Score: ', score[1]*100)