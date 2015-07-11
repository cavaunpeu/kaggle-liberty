import os
import pickle

from utils import DATA_DIR


class Dataset(object):

    def __init__(self, func, finalized=True, **kwargs):
        self.func = func
        self.kwargs = kwargs
        self.finalized = finalized

    def generate(self):
        if self.finalized:
            data_func_dir = DATA_DIR + '/' + self.func.__name__ + self._stringify_kwargs()
            if not os.path.exists(data_func_dir):
                os.mkdir(data_func_dir)
            if all([os.path.exists(data_func_dir + '/' + dataset) for dataset in ['X_train.pkl', 'y_train.pkl', 'X_test.pkl']]):
                print('Loading saved dataset ..')
                X_train = pickle.load(open(data_func_dir + '/X_train.pkl', 'rb'))
                y_train = pickle.load(open(data_func_dir + '/y_train.pkl', 'rb'))
                X_test = pickle.load(open(data_func_dir + '/X_test.pkl', 'rb'))
            else:
                X_train, y_train, X_test = self.func(**self.kwargs)
                pickle.dump(X_train, open(data_func_dir + '/X_train.pkl', 'wb'))
                pickle.dump(y_train, open(data_func_dir + '/y_train.pkl', 'wb'))
                pickle.dump(X_test, open(data_func_dir + '/X_test.pkl', 'wb'))
        else:
            X_train, y_train, X_test = self.func(**self.kwargs)

        return X_train, y_train, X_test

    def _stringify_kwargs(self):
        ret = ''
        for k, v in self.kwargs.iteritems():
            if hasattr(v, '__name__'):
                ret += '_' + str(k) + '_' + v.__name__ + '_' + str(v)
            elif hasattr(v.__class__, '__name__'):
                ret += '_' + (str(k) + '_' + v.__class__.__name__ + '_' + str(v))
            else:
                ret += '_' + append(str(k) + '_' + str(v))
        return ret