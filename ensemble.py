import glob
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.optimize import fmin_powell

from utils import Y_TRAIN, DATA_DIR, SUBMISSION_PATH
from utils import load_predictions_with_cutoff, normalized_gini, find_ensemble_weights, \
    _ensemble_predictions


class Ensemble(object):

    def __init__(self, base_path, opt_func=fmin_powell, cutoff=.38, include=[], verbose=True):
        self.opt_func = opt_func
        self.base_path = base_path
        self.cutoff = cutoff
        self.include = include
        self.verbose = verbose
        self.oof_predictions, self.lb_predictions, self.oof_ginis = load_predictions_with_cutoff(
            base_path, cutoff, include)
        self.opt_weights = None
        self.ensembled_oof_predictions_ = None
        self.ensembled_oof_gini_ = None
        self.ensembled_lb_predictions_ = None

    def ensemble_predictions(self):
        # find optimal weights
        self.opt_weights = find_ensemble_weights(
            opt_func=self.opt_func,
            predictions=self.oof_predictions,
            y_true=Y_TRAIN,
            w_init=np.array(self.oof_ginis)**0.5,
            verbose=self.verbose
        )['x']

        # ensemble oof predictions
        self.ensembled_oof_predictions_ = _ensemble_predictions(
            predictions=self.oof_predictions,
            ensemble_weights=self.opt_weights
        )
        self.ensembled_oof_gini_ = normalized_gini(Y_TRAIN, self.ensembled_oof_predictions_)
        if self.verbose:
            print '\nEnsembled CV: {}\n'.format(np.round(self.ensembled_oof_gini_, 6))
            for w, pred_col in zip(self.opt_weights, self.oof_predictions.columns):
                print '{} : {}'.format(np.round(w, 4), pred_col)

        # ensemble lb predictions
        self.ensembled_lb_predictions_ = _ensemble_predictions(
            predictions=self.lb_predictions,
            ensemble_weights=self.opt_weights
        )

    def create_submission(self, sub_path):
        sub = pd.read_csv(DATA_DIR + '/sample_submission.csv', index_col='Id')
        sub_path = SUBMISSION_PATH + '/' + sub_path.replace('.csv', '')
        sub_path += '_CV_{}_.csv'.format(str(np.round(self.ensembled_oof_gini_, 6)))
        sub['Hazard'] = self.ensembled_lb_predictions_
        sub.to_csv(sub_path, index='Id')
