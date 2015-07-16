import glob
import hashlib
import pickle
from pprint import pprint
import os

import numpy as np
import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
TRAIN_PATH = DATA_DIR + '/train.csv'
TEST_PATH = DATA_DIR + '/test.csv'
PREDICTION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'predictions')
SUBMISSION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'submissions')
Y_TRAIN = pd.read_csv(TRAIN_PATH)['Hazard'].values


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


def print_prediction_metadata(path=None):
    if path:
        pred = pickle.load(open(PREDICTION_PATH + '/' + path, 'rb'))
        print 'Dataset func: {}'.format(pred['dataset_func'])
        print 'Dataset func params: {}'.format(pred['dataset_params'])
        print 'Model name: {}'.format(pred['model_name'])
        print 'Model params: {}'.format(pprint(pred['model_params']))
        print 'Target transform: {}'.format(pred['target_transform'])
        print 'Feature selector: {}'.format(pred['feature_selector'])
        print 'Feature selector params: {}'.format(pred['feature_selector_params'])
        print 'CV: {}\n'.format(np.round(pred['normalized_gini'], 4))
    else:
        for p in glob.glob(PREDICTION_PATH + '/*'):
            try:
                pred = pickle.load(open(p, 'rb'))
                oof_gini = pred['normalized_gini']
                print '`%s` | CV: %s' % (p.split('/')[-1], str(np.round(oof_gini, 4)))
            except EOFError:
                continue
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


def load_predictions_with_cutoff(base_path, cutoff, include=[], verbose=True):
    oof_predictions, lb_predictions, names, oof_ginis = [], [], [], []

    for f in glob.glob(base_path + '/*'):
        try:
            pred = pickle.load(open(f, 'rb'))
            oof_gini = np.round(pred['normalized_gini'], 5)
            pred_name = str(f.split('/')[-1].split('.')[0])
            path_ = f.split('/')[-1]
            if oof_gini > cutoff or path_ in include:
                if verbose:
                    print '{} | CV: {}'.format(pred_name, oof_gini)
                oof_predictions.append(pred['oof_predictions'])
                lb_predictions.append(pred['lb_predictions'])
                names.append(pred_name)
                oof_ginis.append(pred['normalized_gini'])
        except EOFError:  # 0 size file
            continue

    oof_predictions = pd.DataFrame(oof_predictions).T
    oof_predictions.columns = names
    lb_predictions = pd.DataFrame(lb_predictions).T
    lb_predictions.columns = names

    if verbose:
        print '\n{} predictions loaded.'.format(str(len(names)))

    return oof_predictions, lb_predictions, oof_ginis

def _ensemble_predictions(predictions, ensemble_weights):
    ensembled_predictions = np.zeros(shape=(predictions.shape[0],))
    for w, pred_col in zip(ensemble_weights, predictions.columns):
        ensembled_predictions += w*predictions[pred_col]
    return ensembled_predictions

def find_ensemble_weights(opt_func, predictions, y_true, w_init=None, verbose=True):

    def normalized_gini_func(weights):
        ensembled_predictions = _ensemble_predictions(predictions, weights)
        ensembled_oof_gini = normalized_gini(y_true, ensembled_predictions)
        return -ensembled_oof_gini

    def equality_constraint(weights):
        return sum(weights) - 1

    if w_init is None:
        w_init = [0.01]*len(predictions)

    return opt_func(normalized_gini_func, w_init, constraints={'type':'eq', 'fun': equality_constraint})
    # return opt_func(normalized_gini_func, w_init)

