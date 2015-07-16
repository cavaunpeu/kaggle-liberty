import inspect
import os
import pickle
from types import FunctionType

import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold


from dataset import Dataset
from feature_selection import *
from target_transform import *
from utils import md5, touch, normalized_gini


class Prediction(object):

    '''Prediction object for Liberty Insurance Competition.
    '''

    def __init__(self, dataset, model, base_path, target_transform=BaseTargetTransform,
            feature_selector=BaseFeatureSelector(), save=True):
        self.dataset = dataset
        self.model = model
        self.target_transform = target_transform
        self.feature_selector = feature_selector
        self.save = save
        self.save_path = self._generate_filename(base_path)

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

    def _get_dataset_func_name(self):
        return self.dataset.func.__name__

    def _get_feature_selector_name(self):
        return self.feature_selector.__class__.__name__

    def _get_target_transform_name(self):
        return self.target_transform.__name__

    @staticmethod
    def _get_init_args_from_class(my_class):
        init_args = inspect.getargspec(my_class.__init__).args[1:]
        init_args_dict = {a: my_class.__dict__[a] for a in init_args}
        return init_args_dict

    def _hash_model_params(self):
        params = self._extract_model_parameters()
        params_str = ' '.join(['%s=%s' % (k, v) for k, v in params.items()])
        return md5(params_str)

    def _hash_feature_selector_params(self):
        params = repr(self._get_init_args_from_class(self.feature_selector))
        return md5(params)

    def _hash_function_kwargs(self):
        ret = []
        for k, v in self.dataset.kwargs.iteritems():
            if type(v) == FunctionType:
                ret.append(str(k) + '_' + v.__name__)
            elif hasattr(v, '__name__'):
                ret.append(str(k) + '_' + v.__name__ + '_' + str(v))
            elif hasattr(v.__class__, '__name__'):
                ret.append(str(k) + '_' + v.__class__.__name__ + '_' + str(v))
            else:
                ret.append(str(k) + '_' + str(v))

        return md5(''.join(ret))

    def _generate_filename(self, base_path):
        filename_args = {
            'dataset_func': self._get_dataset_func_name(),
            'hashed_func_kwargs': self._hash_function_kwargs(),
            'model_name': self._get_model_name(),
            'hashed_model_params': self._hash_model_params(),
            'target_transform_name': self._get_target_transform_name(),
            'feature_selector_name': self._get_feature_selector_name(),
            'hashed_feature_selector_params': self._hash_feature_selector_params()
        }

        filename = '%(dataset_func)s_%(hashed_func_kwargs)s_%(model_name)s_%(hashed_model_params)s_%(target_transform_name)s_%(feature_selector_name)s_%(hashed_feature_selector_params)s.pkl' % filename_args
        return os.path.join(base_path, filename)

    def _generate_oof_predictions(self, X_train, y_train, feature_selection_func=None, **kwargs):
        train_n = X_train.shape[0]
        cv = KFold(n=train_n, n_folds=10, random_state=123)
        cv_scores = []

        oof_predictions = np.zeros(shape=(train_n,))
        for fold, (tr_idx, te_idx) in enumerate(cv):

            X_train_ = X_train.iloc[tr_idx]
            y_train_ = self.target_transform.transform(y_train.iloc[tr_idx])
            X_test_  = X_train.iloc[te_idx]
            y_test_  = y_train.iloc[te_idx]

            # select features
            self.feature_selector.fit(X_train_, y_train_)
            X_train_ = self.feature_selector.transform(X_train_)
            X_test_  = self.feature_selector.transform(X_test_)


            self.model.fit(X_train_, y_train_)
            preds_k = self.target_transform.transform_back(self.model.predict(X_test_))
            oof_predictions[te_idx] = preds_k

            gini_k = normalized_gini(y_test_, preds_k)
            cv_scores.append(gini_k)
            print 'Fold %d: %.4f' % (fold + 1, gini_k)

        oof_gini = normalized_gini(y_train, oof_predictions)
        print 'Final: %.4f' % (oof_gini)
        return oof_predictions, oof_gini, cv_scores

    def _generate_lb_predictions(self, X_train, y_train, X_test):
        train_n, test_n = X_train.shape[0], X_test.shape[0]
        index = ['train']*train_n + ['test']*test_n
        predictions = pd.DataFrame(np.zeros(shape=(train_n + test_n,)), index=index)

        # select features
        final_features = self.feature_selector.return_final_features()
        X_train, X_test = X_train[final_features], X_test[final_features]

        self.model.fit(X_train, self.target_transform.transform(y_train))
        lb_predictions = self.target_transform.transform_back(self.model.predict(X_test))
        return lb_predictions

    def cross_validate(self, feature_selection_func=None, **kwargs):
        if self.save is True:
            if os.path.exists(self.save_path):
                try:
                    pred = pickle.load(open(self.save_path, 'rb'))
                    oof_gini = pred['normalized_gini']
                    print '`{}` exists. CV: {}'.format(
                        self.save_path.split('/')[-1],
                        np.round(oof_gini, 4)
                    )
                    return
                except EOFError:
                    pass

            # reserve the filename
            touch(self.save_path)

        X_train, y_train, X_test = self._load_data()
        oof_predictions, oof_gini, cv_scores = self._generate_oof_predictions(X_train, y_train)
        lb_predictions = self._generate_lb_predictions(X_train, y_train, X_test)

        # save
        if self.save is True:
            to_save = {
                'oof_predictions': oof_predictions,
                'lb_predictions': lb_predictions,
                'normalized_gini': oof_gini,
                'normalized_gini_cv': cv_scores,
                'model_params': self._extract_model_parameters(),
                'model_name': self._get_model_name(),
                'dataset_func': self._get_dataset_func_name(),
                'dataset_params': self.dataset.kwargs,
                'target_transform': self._get_target_transform_name(),
                'feature_selector': self._get_feature_selector_name(),
                'feature_selector_params': self._get_init_args_from_class(self.feature_selector)
            }
            pickle.dump(to_save, open(self.save_path, 'wb'))
