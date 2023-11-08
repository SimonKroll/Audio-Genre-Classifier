'''
much credit goes to https://github.com/ruohoruotsi/LSTM-Music-Genre-Classification/blob/master/lstm_genre_classifier_keras.py for most of the code
Modifications by Ethan B for 477 course project

'''

import logging
import os, path
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.optimizers import Adam
from keras import initializers
from keras import backend as K
from keras.models import model_from_json
from GenreFeatureData import (
    GenreFeatureData,
)  # local python class with Audio feature extraction (librosa)

# set logging level
logging.getLogger("tensorflow").setLevel(logging.ERROR)

genre_features = GenreFeatureData()

# if all of the preprocessed files do not exist, regenerate them all for self-consistency
if (
    os.path.isfile(genre_features.train_X_preprocessed_data)
    and os.path.isfile(genre_features.train_Y_preprocessed_data)
    and os.path.isfile(genre_features.test_X_preprocessed_data)
    and os.path.isfile(genre_features.test_Y_preprocessed_data)
):
    print("Preprocessed files exist, deserializing npy files")
    genre_features.load_deserialize_data()
else:
    print("Preprocessing raw audio files")
    genre_features.load_preprocess_data()

print("Training X shape: " + str(genre_features.train_X.shape))
print("Training Y shape: " + str(genre_features.train_Y.shape))

print("Test X shape: " + str(genre_features.test_X.shape))
print("Test Y shape: " + str(genre_features.test_Y.shape))

input_shape = (genre_features.train_X.shape[1], genre_features.train_X.shape[2])
print("Build LSTM RNN model ...")
model = Sequential()

initializer = initializers.RandomNormal(mean=0., stddev=1.)

lstm0 = LSTM(kernel_initializer=initializer, units=64, recurrent_dropout=0.20, return_sequences=True, input_shape=input_shape)
model.add(lstm0)
lstm1 = LSTM(kernel_initializer=initializer, units=24, recurrent_dropout=0.20, return_sequences=True)
model.add(lstm1)
lstm2 = LSTM(kernel_initializer=initializer, units=16, recurrent_dropout=0.20, return_sequences=False)
model.add(lstm2)
dense0 = Dense(kernel_initializer=initializer, units=genre_features.train_Y.shape[1], activation="softmax")
model.add(dense0)

print("Compiling ...")

opt = Adam(learning_rate=0.01, epsilon=1e-8, weight_decay=0.001)
model.compile(loss="MSE", optimizer=opt, metrics=["accuracy"])
model.summary()

print("Training ...")
batch_size = 30  # num of training examples per minibatch
num_epochs = 100
model.fit(
    genre_features.train_X,
    genre_features.train_Y,
    batch_size=batch_size,
    epochs=num_epochs,
)

print("\nTesting ...")
score, accuracy = model.evaluate(
    genre_features.test_X, genre_features.test_Y, batch_size=batch_size, verbose=1
)
print("Test loss:  ", score)
print("Test accuracy:  ", accuracy)


# Creates a HDF5 file 'lstm_genre_classifier.h5'
model_filename = "lstm_genre_classifier_model.h5"
print("\nSaving model: " + model_filename)
model.save("model/" + model_filename)
# Creates a json file
print("creating .json file....")
model_json = model.to_json()
f = path.Path("model/lstm_genre_classifier_model.json")
f.write_text(model_json)
