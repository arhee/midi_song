from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file

import mido
import numpy as np
from math import ceil
import random
import music21
import glob
import os
import pickle

def get_notelist(mido_obj, res=8, note_range=(36,84)):
    """
    Args: 
        res (int) - resolution.  8 is eighth notes, 4 quarter, etc...
        note_range (tuple) - middle C = 60
    """

    # this gives # of ticks in a column of the piano sheet
    tick_step = round(float(mido_obj.ticks_per_beat/res))

    active_notes = [0] * abs(np.diff(note_range)[0])

    counter = 0
    notelist = []

    rawmsgs = [msg for track in mido_obj.tracks for msg in track]

    # have to reset counter when a new track is detected
    for msg in rawmsgs:
        if msg.type not in set(['note_on','note_off']):
            continue

        #counter += ceil(msg.time/tick_step)
        counter += msg.time

        if msg.type == 'note_on' and msg.velocity > 0:        
            active_notes[msg.note - note_range[0]] = counter

        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            # fill everything up to this with 1s
            start = active_notes[msg.note - note_range[0]]  #round this section

            notelist.append((msg.note, int(ceil(start/tick_step)), int(ceil((counter-start)/tick_step))))
            active_notes[msg.note - note_range[0]] = 0

    notelist = sorted(notelist, key=lambda x: x[1])
    return (counter, notelist)


def make_musicmat(notelist, steps, note_range=(36,84)):
    music_mat = np.zeros([steps, int(np.diff(note_range))])
    cnt = 0
    for nt in notelist:
        c = range(nt[1], (nt[2] + nt[1]))
        r = [nt[0] - note_range[0]] * len(c)
        music_mat[c,r] = np.array([1] * len(r) )
        cnt += 1
    return music_mat

def segmentize(data, maxlen, step):
    X = []
    y = []
    steps = data.shape[0]
    for ix in range(0, steps-maxlen-1, step):
        end = ix + maxlen
        X.append(data[ix:end:1, :])
        y.append(data[end+1,:])
    X = np.stack(X)
    y = np.stack(y)
    return X,y

################################################
#params
maxlen = 50
res = 8
note_range=(36,84)
nnotes = np.diff(note_range)[0]
step = 1
################################################

raw_data = []
guitar_dir = 'data/tr_guitar_licks/'
for fname in glob.glob(guitar_dir + '*'):
    try:
        midifile = mido.MidiFile(fname)
        counter, notelist = get_notelist(midifile)
        tick_step = round(float(midifile.ticks_per_beat/res))
        steps = int(ceil(counter/tick_step))
        music_mat = make_musicmat(notelist, steps)
        raw_data.append(music_mat)
    except:
        pass

X = []
y = []
for sample in raw_data:
    smpX, smpy = segmentize(sample, maxlen, 1)
    X.append(smpX)
    y.append(smpy)

X = np.concatenate(X, axis=0)
y = np.concatenate(y, axis=0)        


model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, nnotes)))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dense(nnotes))
model.compile(loss='categorical_crossentropy', optimizer='adam')


model.fit(X,y, batch_size=100, nb_epoch=10)

model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

model.save_weights('model.h5')

with open('song_mat.p','wb') as f:
    pickle.dump(X, f)
