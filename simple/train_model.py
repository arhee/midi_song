"""
Trains model and saves the model params
"""

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
import pickle

# load the data
with open('mmat.p') as f:
    data = pickle.load(f)

X = data['X']
y = data['y']
note_range = data['note_range']


model = Sequential()
model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dense(X.shape[1]))
model.compile(loss='categorical_crossentropy', optimizer='adam')


model.fit(X,y, batch_size=32, nb_epoch=10)

model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

model.save_weights('model.h5')