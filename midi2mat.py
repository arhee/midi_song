""" Coverts all midi files in a directory to a musicmat object in a pickle file
"""

import mido
import glob
import numpy as np
import argparse
import pickle
import sys
from math import ceil
from scipy import sparse
from utils import *

def main():
    # params
    # beat resolution -> 8: 8th note, 4: qtr note
    res = 8
    note_range = (36, 84)

    raw_data = []

    guitar_dir = opts.dir
    print 'loading data'
    for fname in glob.glob(guitar_dir + '*'):
        try:
            midifile = mido.MidiFile(fname)
            counter, notelist = get_notelist(midifile)
            tick_step = round(float(midifile.ticks_per_beat / res))
            steps = int(ceil(counter / tick_step))
            music_mat = make_musicmat(notelist, steps)
            raw_data.append(music_mat)
        except IOError:
             pass

    maxlen = 50

    X = []
    y = []
    for sample in raw_data:
        smpX, smpy = segmentize(sample, maxlen, 1)
        X.append(smpX)
        y.append(smpy)

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)

    with open(opts.outfile,'wb') as f:
        print 'dumping matrices'
        pickle.dump({'X': X, 'y':y}, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts midi files to a matrix')
    parser.add_argument('-d', '--dir', action='store', required=True,
        help='Set destination -- backstage, prospecting')
    parser.add_argument('-o', '--outfile', action='store', required=True,
        help='Set user authorization.')

    opts, args = parser.parse_known_args(sys.argv[1:])
    main()
