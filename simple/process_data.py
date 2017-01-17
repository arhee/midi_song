"""  A simplified version of the problem that only looks at note
sequence

generates a matrix of note sequences.

"""

import mido
import glob
import numpy as np
from math import ceil
import pickle
import numpy as np

def get_notelist(mido_obj, res=16, note_range=(36,84)):
    """
    Args:
        res (int) - resolution.  8 is eighth notes, 4 quarter, etc...
        note_range (tuple) - middle C = 60
    Returns:
        counter (int) - total amount of time elapsed for song
        notelist (list) - a list of notes in (pitch, start, duration) format
    """

    # this gives # of ticks in a column of the piano sheet
    tick_step = round(float(mido_obj.ticks_per_beat/res))

    active_notes = [0] * abs(np.diff(note_range)[0])

    counter = 0
    notelist = []

    # this needs to be fixed in the future for multiinstrument songs
    rawmsgs = [msg for track in mido_obj.tracks for msg in track]

    # have to reset counter when a new track is detected
    for msg in rawmsgs:
        if msg.type not in set(['note_on','note_off']):
            continue

        #counter += ceil(msg.time/tick_step)
        counter += msg.time

        if msg.type == 'note_on' and msg.velocity > 0:
            try:
                active_notes[msg.note - note_range[0]] = counter
            except IndexError:
                pass

        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            # fill everything up to this with 1s
            try:
                start = active_notes[msg.note - note_range[0]]  #round this section
                notelist.append((msg.note, int(ceil(start/tick_step)), int(ceil((counter-start)/tick_step))))
                active_notes[msg.note - note_range[0]] = 0
            except IndexError:
                pass

    notelist = sorted(notelist, key=lambda x: x[1])
    return (counter, notelist)


mididir = '../data/midi/tr_guitar_licks/'
midifnames = glob.glob(mididir + '*')

midifiles = []
for x in midifnames:
    try:
        midifiles.append(mido.MidiFile(x))
    except IOError:
        pass

# get the order of notes
data = []
for obj in midifiles:
    counter, notelist = get_notelist(obj)
    data.append([x[0] for x in notelist])

# shrink the dataset by reducing by min note
min_note = min([min(y) for y in data])
max_note = max([max(y) for y in data])
normdata = [np.array(x) - min_note for x in data]
delta = max_note - min_note

# create matrix
mat_data = []
for x in normdata:
    # rows, cols
    mat = np.zeros((delta+1, len(x)))
    mat[x,range(len(x))] = 1
    mat_data.append(mat)

# segmentize
# generate pickled data
X = []
y = []

maxlen = 10
step = 1

for obj in mat_data:
    steps = obj.shape[1]
    # rows cols
    for ix in range(0, steps-maxlen-1, step):
        end = ix + maxlen
        X.append(obj[:, ix:end:1])
        y.append(obj[:, end+1])
X = np.stack(X)
y = np.stack(y)


data = {'X':X, 'y':y, 'note_range':delta}

with open('mmat.p','wb') as f:
    pickle.dump(data, f)