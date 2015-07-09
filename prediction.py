import os
import pickle

import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold, StratifiedKFold
from types import FunctionType

from dataset import Dataset
from utils import md5, touch, normalized_gini, \
    PREDICTION_PATH, SUBMISSION_PATH, DATA_DIR


class Prediction(object):

    '''Prediction object for Crowdflower Search Results Relevance.
    '''

    def __init__(self, dataset, model, save=True):
        self.dataset = dataset
        self.model = model
        self.save = save
        self.oof_predictions = None
        self.oof_gini = None
        self.lb_predictions = None

    def _load_data(self):
        X_train, y_train, X_test = self.dataset.generate()
        return X_train, y_train, X_test

    def _extract_model_parameters(self):
        if hasattr(self.model, 'estimator_params'):
            return dict((param, getattr(self.model, param)) for param in
                        self.model.estimator_params)
        elif hasattr(self.model, '_get_param_names'):
            return dict((param, getattr(self.model, param)) for param in
                        self.model._get_param_names())
        else:
            raise NotImplemented, 'model must implement estimator_params' \
                                  ' or _get_param_names()'

    def _get_model_name(self):
        return self.model.__class__.__name__

    def _hash_model_params(self):
        params = self._extract_model_parameters()
        params_str = ' '.join(['%s=%s' % (k, v) for k, v in params.items()])
        return md5(params_str)

    @staticmethod
    def _hash_function_kwargs(kwargs):
        ret = []
        for k, v in kwargs.iteritems():
            if type(v) == FunctionType:
                ret.append(str(k) + '_' + v.__name__)
            elif hasattr(v, '__name__'):
                ret.append(str(k) + '_' + v.__name__ + '_' + str(v))
            elif hasattr(v.__class__, '__name__'):
                ret.append(str(k) + '_' + v.__class__.__name__ + '_' + str(v))
            else:
                ret.append(str(k) + '_' + str(v))

        if len(ret) == 0:
            return ''
        else:
            return md5(''.join(ret))

    def _generate_filename(self):
        filename_args = {
            'dataset_func': self.dataset.func.__name__,
            'hashed_func_kwargs': self._hash_function_kwargs(self.dataset.kwargs),
            'model_name': self._get_model_name(),
            'hashed_model_params': self._hash_model_params()
        }

        filename = '%(dataset_func)s_%(hashed_func_kwargs)s%(model_name)s_%(hashed_model_params)s.pkl' % filename_args
        return os.path.join(PREDICTION_PATH, filename)

    def cross_validate(self):
        if self.save is True:
            save_path = self._generate_filename()
            if os.path.exists(save_path):
                print 'Model %s exists. Skipping ..' % save_path
                return

            # reserve the filename
            touch(save_path)

        X_train, y_train, X_test = self._load_data()
        train_n = X_train.shape[0]
        test_n = X_test.shape[0]
        n = train_n + test_n

        # create oof predictions
        cv = KFold(n=train_n, n_folds=10, random_state=123)
        cv_scores = []

        oof_predictions = np.zeros(shape=(train_n,))
        for fold, (tr_idx, te_idx) in enumerate(cv):

            X_train_ = X_train.iloc[tr_idx]
            y_train_ = y_train.iloc[tr_idx]

            self.model.fit(X_train_, y_train_)
            preds_k = self.model.predict(X_train.iloc[te_idx])
            oof_predictions[te_idx] = preds_k

            gini_k = normalized_gini(y_train.iloc[te_idx], preds_k)
            cv_scores.append(gini_k)
            print 'Fold %d: %.4f' % (fold + 1, gini_k)

        oof_gini = normalized_gini(y_train, oof_predictions)
        print 'Final: %.4f' % (oof_gini)
        self.oof_predictions = oof_predictions
        self.oof_gini = oof_gini

        # fit model on all data, create leaderboard predictions
        index = ['train']*train_n + ['test']*test_n
        predictions = pd.DataFrame(np.zeros(shape=(n,)), index=index)

        self.model.fit(X_train, y_train)
        lb_predictions = self.model.predict(X_test)
        self.lb_predictions = lb_predictions

        # save
        if self.save is True:
            to_save = {
                'oof_predictions': oof_predictions,
                'lb_predictions': lb_predictions,
                'normalized_gini': oof_gini,
                'normalized_gini_cv': cv_scores,
                'model_params': self._extract_model_parameters(),
                'model_name': self._get_model_name(),
                'dataset_params': self.dataset.kwargs
            }
            pickle.dump(to_save, open(save_path, 'wb'))

    def create_submission(self, file_name):
        path = SUBMISSION_PATH + '/{}_{}.csv'.format(file_name, str(np.round(self.oof_gini, 2)))
        sub = pd.read_csv(DATA_DIR + '/sample_submission.csv', index_col='Id')
        sub['Hazard'] = self.lb_predictions
        sub.to_csv(path, index='Id')
