import argparse
import pickle
from models import *

def main():
    with open(opts.data) as f:
        data = pickle.load(f)

    X = data['X']
    y = data['y']

    model = default_model()

    model.fit(X,y, batch_size=50, nb_epoch=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains model on data')
    parser.add_argument('-d', '--data', action='store', required=True,
        help='pickle data to train on')
    parser.add_argument('-o', '--outfile', action='store', required=True,
        help='filename for model params')

    opts, args = parser.parse_known_args(sys.argv[1:])
    main()
