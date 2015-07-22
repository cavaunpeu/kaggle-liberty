import abc
from datetime import datetime
import glob
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.optimize import fmin_powell

from utils import Y_TRAIN, DATA_DIR, SUBMISSION_PATH
from utils import load_predictions_with_cutoff, normalized_gini, find_ensemble_weights, \
    _ensemble_predictions


class BaseEnsemble(object):

    __metaclass__ = abc.ABCMeta

    ensembled_oof_predictions_ = None
    ensembled_oof_gini_ = None
    ensembled_lb_predictions_ = None

    @abc.abstractmethod
    def ensemble_predictions():
        return

    def create_submission(self, sub_path):
        sub = pd.read_csv(DATA_DIR + '/sample_submission.csv', index_col='Id')
        sub_path = SUBMISSION_PATH + '/' + sub_path.replace('.csv', '')
        current_datetime = datetime.now().strftime('%Y%m%d%H%M')
        gini_str = str(np.round(self.ensembled_oof_gini_, 6))
        sub_path += '_{}_CV_{}.csv'.format(current_datetime, gini_str)
        sub['Hazard'] = self.ensembled_lb_predictions_.values
        sub.to_csv(sub_path, index='Id')


class WeightedEnsemble(BaseEnsemble):

    def __init__(self, base_path, opt_func=fmin_powell, cutoff=.38, column_picker_func=None,
            include=[], verbose=True, **kwargs):
        self.opt_func = opt_func
        self.base_path = base_path
        self.cutoff = cutoff
        self.include = include
        self.verbose = verbose
        self.kwargs = kwargs
        self.oof_predictions, self.lb_predictions, self.oof_ginis = load_predictions_with_cutoff(
            base_path, cutoff, include)
        if column_picker_func is not None:
            new_cols = column_picker_func(self.oof_predictions)
            self.oof_predictions = self.oof_predictions[new_cols]
            self.lb_predictions = self.lb_predictions[new_cols]
        self.opt_weights = None

    def ensemble_predictions(self):
        # find optimal weights
        self.opt_weights = find_ensemble_weights(
            opt_func=self.opt_func,
            predictions=self.oof_predictions,
            y_true=Y_TRAIN,
            w_init=np.array(self.oof_ginis)**0.5,
            verbose=self.verbose,
            **self.kwargs
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


class FunctionalEnsemble(BaseEnsemble):

    def __init__(self, base_path, ensemble_func, cutoff=.38, column_picker_func=None,
            include=[], verbose=True, **column_picker_kwargs):
        self.ensemble_func = ensemble_func
        self.base_path = base_path
        self.cutoff = cutoff
        self.include = include
        self.verbose = verbose
        self.oof_predictions, self.lb_predictions, self.oof_ginis = load_predictions_with_cutoff(
            base_path, cutoff, include)
        if column_picker_func is not None:
            new_cols = column_picker_func(self.oof_predictions, **column_picker_kwargs)
            self.oof_predictions = self.oof_predictions[new_cols]
            self.lb_predictions = self.lb_predictions[new_cols]

    def ensemble_predictions(self):
        # ensemble oof predictions
        self.ensembled_oof_predictions_ = self.oof_predictions.apply(self.ensemble_func, axis=1)
        self.ensembled_oof_gini_ = normalized_gini(Y_TRAIN, self.ensembled_oof_predictions_)
        if self.verbose:
            print '\nEnsembled CV: {}\n'.format(np.round(self.ensembled_oof_gini_, 6))

        # ensemble lb predictions
        self.ensembled_lb_predictions_ = self.lb_predictions.apply(self.ensemble_func, axis=1)
