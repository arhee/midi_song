""" Utilities to help with midi manipulation
"""

import numpy as np
from math import ceil

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
    """ turns a notelist into a musicmat object
    """
    music_mat = np.zeros([steps, int(np.diff(note_range))])
    cnt = 0
    for nt in notelist:
        c = range(nt[1], (nt[2] + nt[1]))
        r = [nt[0] - note_range[0]] * len(c)
        music_mat[c,r] = np.array([1] * len(r) )
        cnt += 1
    return music_mat

def segmentize(musicmat, maxlen, step):
    """ Converts a musicmat matrix into
    a 3D matrix with slices of length maxlen
    """
    X = []
    y = []
    steps = musicmat.shape[0]
    for ix in range(0, steps-maxlen-1, step):
        end = ix + maxlen
        X.append(musicmat[ix:end:1, :])
        y.append(musicmat[end+1,:])
    X = np.stack(X)
    y = np.stack(y)
    return X,y

