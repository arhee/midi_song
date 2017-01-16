import argparse
import pickle
from models import *

tsteps = 100

start_ix = random.randint(0, steps-maxlen - 1)
psg_seed = music_mat[start_ix:start_ix+maxlen,:]
x = psg_seed[None,:,:]


new_song = []
new_song.append(x)

for _ in range(tsteps):

    preds = model.predict(x, verbose=0)[0]

    # this is a timestep slice
    new_note = np.zeros(nnotes)
    new_note[preds.argmax()] = 1
    new_note = new_note[None, None,:]

    # redo the seed psg
    psg = np.concatenate([x,new_note], axis=1)
    x = psg[:, 1:,:]

    new_song.append(new_note)

new_song = np.concatenate(new_song, axis=1)
new_song = np.squeeze(new_song)


def main():
    with open(opts.data) as f:
        data = pickle.load(f)

    X = data['X']
    y = data['y']

    train_len = X.shape[1]
    model = default_model(train_len)
    model.fit(X,y, batch_size=50, nb_epoch=1)

    print 'saving data'
    with open(opts.outfile + '.json', 'w') as json_file:
        json_file.write(model.to_json())
    model.save_weights(opts.outfile + '.h5')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains model on data')
    parser.add_argument('-d', '--data', action='store', required=True,
        help='pickle data to train on')
    parser.add_argument('-o', '--outfile', action='store', required=True,
        help='filename for model params')

    opts, args = parser.parse_known_args(sys.argv[1:])
    main()
