import argparse
import pickle
from models import *

def main():
    with open(opts.data_file) as f:
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
    parser.add_argument('-d', '--data_file', action='store', required=True,
        help='pickle data to train on')
    parser.add_argument('-o', '--outfile', action='store', required=True,
        help='filename for model params')

    opts, args = parser.parse_known_args(sys.argv[1:])
    main()
