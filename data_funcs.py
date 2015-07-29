import pandas as pd
from sklearn.preprocessing import scale

from kaggle_tools.features import encode_with_observation_counts, encode_with_leave_one_out
from utils import load_data, load_predictions_with_cutoff, PREDICTION_PATH, \
    least_correlated_cols



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


def data_v5():
    X, y_train = load_data()
    X = pd.get_dummies(X)
    X.drop(['T2_V10', 'T2_V7', 'T1_V13', 'T1_V10'], axis=1, inplace=True)

    is_train_obs = X.index.get_level_values('obs_type') == 'train'
    X_train, X_test = X[is_train_obs], X[~is_train_obs]
    return X_train, y_train, X_test


def data_v6():
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

    cols_to_drop = ['T2_V10', 'T2_V7', 'T1_V13', 'T1_V10']
    X_train.drop(['T2_V10', 'T2_V7', 'T1_V13', 'T1_V10'], axis=1, inplace=True)
    X_test.drop(['T2_V10', 'T2_V7', 'T1_V13', 'T1_V10'], axis=1, inplace=True)
    return X_train, y_train, X_test


def stacker_data_v1(cutoff):
    X, y_train = load_data()
    oof_predictions, lb_predictions, oof_ginis = load_predictions_with_cutoff(PREDICTION_PATH, cutoff)
    X_train, X_test = oof_predictions, lb_predictions
    return X_train, y_train, X_test

def stacker_data_v2(cutoff, num_least_correlated_cols):
    X, y_train = load_data()
    oof_predictions, lb_predictions, oof_ginis = load_predictions_with_cutoff(PREDICTION_PATH, cutoff)
    X_train, X_test = oof_predictions, lb_predictions
    new_cols = least_correlated_cols(X_train, num_least_correlated_cols)
    X_train, X_test = X_train[new_cols], X_test[new_cols]
    return X_train, y_train, X_test


