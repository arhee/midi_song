import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
import sys
from keras.utils.data_utils import get_file


def default_model(train_len, note_range=(36,84)):
    """

    :param maxlen: the length of a training sample
    :param note_range:
    :return: keras model
    """
    nnotes = np.diff(note_range)[0]

    model = Sequential()
    model.add(LSTM(128, input_shape=(train_len, nnotes)))
    model.add(Dense(nnotes))
    #model.add(Dense(nnotes))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model