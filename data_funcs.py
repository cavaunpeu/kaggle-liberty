import pandas as pd
from sklearn.preprocessing import scale

from kaggle_tools.features import encode_with_observation_counts, encode_with_leave_one_out
from utils import load_data


def data_v1():
    X, y_train = load_data()
    X = pd.get_dummies(X)

    is_train_obs = X.index.get_level_values('obs_type') == 'train'
    X_train, X_test = X[is_train_obs], X[~is_train_obs]
    return X_train, y_train, X_test


def data_v2(bins, scale_X=False):
    X, y_train = load_data()
    # bin variables that are sinusoidal w.r.t target
    X['T1_V1'] = pd.cut(X['T1_V1'], bins)
    X['T1_V2'] = pd.cut(X['T1_V2'], bins)
    X['T2_V2'] = pd.cut(X['T2_V2'], bins)
    X['T2_V4'] = pd.cut(X['T2_V4'], bins)
    X['T2_V9'] = pd.cut(X['T2_V9'], bins)
    X['T2_V15'] = pd.cut(X['T2_V15'], bins)

    X = pd.get_dummies(X)
    if scale_X is True:
        X = pd.DataFrame(scale(X), columns=X.columns, index=X.index)
    is_train_obs = X.index.get_level_values('obs_type') == 'train'
    X_train, X_test = X[is_train_obs], X[~is_train_obs]
    return X_train, y_train, X_test


def data_v3():
    X, y_train = load_data()
    category_cols = [col for col in X.columns if X[col].dtype == 'O']
    for col in category_cols:
        X[col] = encode_with_observation_counts(X[col])

    is_train_obs = X.index.get_level_values('obs_type') == 'train'
    X_train, X_test = X[is_train_obs], X[~is_train_obs]
    return X_train, y_train, X_test


def data_v4():
    X, y_train = load_data()
    is_train_obs = X.index.get_level_values('obs_type') == 'train'
    X_train, X_test = X[is_train_obs], X[~is_train_obs]

    category_cols = [col for col in X.columns if X[col].dtype == 'O']
    for col in category_cols:
        X_train[col], X_test[col] = encode_with_leave_one_out(
            train_col=X_train[col],
            y=y_train,
            test_col=X_test[col]
        )
    return X_train, y_train, X_test
