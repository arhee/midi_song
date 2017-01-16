import argparse
import pickle
from utils import *
from keras.models import model_from_json
import random
from mido import MidiFile

# load model from disk
# load sample passage
# mat = new_song_mat(model, psg_seed)
# track = notelist_to_track(mat)

def load_model():
    model_dir = 'data/models/'
    model_name = 'test'

    # load json and create model
    with open(model_dir + model_name + '.json', 'r') as json_file:
        loaded_model_json = json_file.read()

    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(model_dir + model_name + '.h5')
    print("Loaded model from disk")

    # evaluate loaded model on test data
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


def load_sample_psg():
    data_fname = 'data/models/transposed_licks.p'
    with open(data_fname) as f:
        data = pickle.load(f)

    X = data['X']

    ix = random.randint(0, X.shape[0])
    psg_seed = X[ix, :, :]
    psg_seed = psg_seed[None, :]
    return psg_seed

model = load_model()
psg = load_sample_psg()

mmat = new_song_mat(model, psg)
track = notelist_to_track(mmat, tick_step=32)
mid = MidiFile()
mid.tracks.append(track)
mid.save('new_song.mid')



# def main():
#     with open(opts.data) as f:
#         data = pickle.load(f)
#
#     X = data['X']
#     y = data['y']
#
#     train_len = X.shape[1]
#     model = default_model(train_len)
#     model.fit(X,y, batch_size=50, nb_epoch=1)
#
#     print 'saving data'
#     with open(opts.outfile + '.json', 'w') as json_file:
#         json_file.write(model.to_json())
#     model.save_weights(opts.outfile + '.h5')
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Trains model on data')
#     parser.add_argument('-d', '--data', action='store', required=True,
#         help='pickle data to train on')
#     parser.add_argument('-m', '--model', action='store', required=True,
#         help='filename for model params')
#     parser.add_argument('-o', '--outfile', action='store', required=True,
#         help='filename for new song')
#
#     opts, args = parser.parse_known_args(sys.argv[1:])
#     main()
#
#
#     # save the file
#     mid = MidiFile()
#     mid.tracks.append(track)
#     mid.save('new_song.mid')