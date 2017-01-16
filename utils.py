""" Utilities to help with midi manipulation
"""

import numpy as np
from math import ceil
from mido import Message, MidiTrack

__all__ = ['get_notelist',
           'make_musicmat',
           'transpose',
           'segmentize',
           'notelist_to_track',
           'new_song_mat']

def get_notelist(mido_obj, res=8, note_range=(36,84)):
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

def make_musicmat(notelist, steps, note_range=(36,84)):
    """ turns a notelist into a musicmat object
    """
    music_mat = np.zeros([steps+1, int(np.diff(note_range))])
    cnt = 0
    for nt in notelist:
        c = range(nt[1], (nt[2] + nt[1]))
        r = [nt[0] - note_range[0]] * len(c)
        try:
            music_mat[c,r] = np.array([1] * len(r) )
        except IndexError:
            pass
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

def transpose(score, newkey='C'):
    """ Transposes the midi file to C.
    Args: music21.stream.Score
    Returns: music21.stream.Score
    """

    keys = {'C':0,
            'C#':1,
            'D-':1,
            'D':2,
            'D#':3,
            'E-':3,
            'E':4,
            'F-':4,
            'E#':5,
            'F':5,
            'F#':6,
            'G-':6,
            'G':7,
            'G#':8,
            'A-':8,
            'A':9,
            'A#':10,
            'B-':10,
            'B':11,
            'B#':0,
           }

    oldkey = score.analyze('key')

    if oldkey.mode == "major":
        halfSteps = keys[oldkey.tonic.name.upper()]

    elif oldkey.mode == "minor":
        halfSteps = keys[oldkey.tonic.name.upper()] + 4

    halfSteps %= 12

    newscore = score.transpose(-halfSteps)
    trans_key = newscore.analyze('key')
    print oldkey, trans_key
    return newscore


def notelist_to_track(new_song, tick_step=100, note_range=(36, 84)):
    """ Converts a notelist to a miditrack.
    Assumes that adjacent notes are held
    """
    track = MidiTrack()
    nsteps = new_song.shape[0]
    last_event = 0
    active_notes = np.zeros(new_song.shape[-1])

    for ix in range(nsteps):
        step_slice = new_song[ix, :]

        diff = [ix for ix, (x, y) in enumerate(zip(step_slice, active_notes)) if x != y]
        if len(diff) == 0:
            last_event += 1
            continue

        # this means there is a difference
        for note_ix in diff:
            # off to on
            note = note_ix + min(note_range)
            if active_notes[note_ix] == 0:
                track.append(Message('note_on', note=note, velocity=127, time=int(tick_step * last_event)))
                active_notes[note_ix] = 1
            else:
                track.append(Message('note_off', note=note, velocity=127, time=int(tick_step * last_event)))
                active_notes[note_ix] = 0
            last_event = 0
    return track


def new_song_mat(model, psg_seed, tsteps=100):
    """ turns a passage seed to a song with #tsteps
    """
    new_song = []
    x = psg_seed
    new_song.append(x)
    nnotes = psg_seed.shape[-1]

    for _ in range(tsteps):
        preds = model.predict(x, verbose=0)[0]

        # this is a timestep slice
        new_note = np.zeros(nnotes)
        new_note[preds.argmax()] = 1
        new_note = new_note[None, None, :]

        # redo the seed psg
        psg = np.concatenate([x, new_note], axis=1)
        x = psg[:, 1:, :]

        new_song.append(new_note)

    new_song = np.concatenate(new_song, axis=1)
    new_song = np.squeeze(new_song)
    return new_song


