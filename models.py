import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
import sys
from keras.utils.data_utils import get_file


def default_model():
    note_range = (36, 84)
    nnotes = np.diff(note_range)[0]
    maxlen = 50

    model = Sequential()
    model.add(LSTM(128, input_shape=(maxlen, nnotes)))
    model.add(Dense(nnotes))
    model.add(Dense(nnotes))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model