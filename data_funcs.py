import pandas as pd

from utils import load_data


def data_v1():
    X, y_train = load_data()
    X = pd.get_dummies(X)

    is_train_obs = X.index.get_level_values('obs_type') == 'train'
    X_train, X_test = X[is_train_obs], X[~is_train_obs]
    return X_train, y_train, X_test
