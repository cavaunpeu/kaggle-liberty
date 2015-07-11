import hashlib
import pickle
from pprint import pprint
import os

import numpy as np
import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TRAIN_PATH = DATA_DIR + '/train.csv'
TEST_PATH = DATA_DIR + '/test.csv'
PREDICTION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'predictions')
SUBMISSION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'submissions')



def md5(s):
    return hashlib.md5(s).hexdigest()


def load_data():
    train = pd.read_csv(TRAIN_PATH, index_col='Id')
    test = pd.read_csv(TEST_PATH, index_col='Id')

    y_train = train['Hazard']
    train.drop('Hazard', axis=1, inplace=True)
    train['obs_type'], test['obs_type'] = 'train', 'test'

    X = pd.concat([train, test]).set_index('obs_type', drop=True, append=True)
    return X, y_train


def touch(fname, times=None):
    with open(fname, 'a'):
        os.utime(fname, times)


def print_prediction_metadata(path):
    pred = pickle.load(open(PREDICTION_PATH + '/' + path, 'rb'))
    print 'Dataset func: {}'.format(pred['dataset_func'])
    print 'Dataset func params: {}'.format(pred['dataset_params'])
    print 'Model name: {}'.format(pred['model_name'])
    print 'Model params: {}'.format(pprint(pred['model_params']))
    print 'CV: {}\n'.format(np.round(pred['normalized_gini'], 4))
    return


def gini(y_true, y_pred):
    df = sorted(zip(y_true, y_pred), key=lambda x: x[1], reverse=True)
    random = [float(i+1)/float(len(df)) for i in range(len(df))]
    totalPos = np.sum([x[0] for x in df])
    cumPosFound = np.cumsum([x[0] for x in df])
    Lorentz = [float(x)/totalPos for x in cumPosFound]
    Gini = [l - r for l, r in zip(Lorentz, random)]
    return np.sum(Gini)


def normalized_gini(y_true, y_pred):
    normalized_gini = gini(y_true, y_pred)/gini(y_true, y_true)
    return normalized_gini
