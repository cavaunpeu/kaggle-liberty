import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from utils import normalized_gini


class BaseFeatureSelector(object):

    selected_features = []
    current_features = []

    def __init__(self):
        return

    def fit(self, X, y):
        self.current_features = X.columns
        self.selected_features += self.current_features

    def transform(self, X):
        return X[self.current_features]

    def return_final_features(self):
        return list(set(self.selected_features))


class RandomForestFeatureSelector(BaseFeatureSelector):

    def __init__(self, top_k_percentile, rf_params):
        self.top_k_percentile = top_k_percentile
        self.model = RandomForestRegressor(**rf_params)

    def fit(self, X, y):
        self.model.fit(X, y)
        feature_importances = [(i, f) for i, f in zip(self.model.feature_importances_, X.columns)]
        feature_importances.sort()
        min_index_to_select = int((1-self.top_k_percentile)*len(feature_importances))
        self.current_features = [f for i, f in feature_importances[min_index_to_select:]]
        self.selected_features += self.current_features


class MinGiniFeatureSelector(BaseFeatureSelector):

    def __init__(self, min_gini):
        self.min_gini = min_gini

    def fit(self, X, y):
        for col in X.columns:
            gini = normalized_gini(y, X[col])
            if abs(gini) > self.min_gini:
                self.current_features.append(col)
        self.selected_features += self.current_features


class TopKPercentileVarianceFeatureSelector(BaseFeatureSelector):

    def __init__(self, top_k_percentile):
        self.top_k_percentile = top_k_percentile

    def fit(self, X, y):
        variances = zip(X.columns, X.apply(np.var, axis=0))
        variances = sorted(variances, key=lambda x: x[1])
        min_index_to_select = int((1-self.top_k_percentile)*len(variances))
        self.current_features = [f for f, v in variances[min_index_to_select:]]
        self.selected_features += self.current_features


class BottomKPercentileVarianceFeatureSelector(BaseFeatureSelector):

    def __init__(self, bottom_k_percentile):
        self.bottom_k_percentile = bottom_k_percentile
        self.current_features = []

    def fit(self, X, y):
        sum_abs_corrs = X.corr().apply(lambda x: sum(abs(x)), axis=1)
        sum_abs_corrs.sort(ascending=False)
        min_index_to_select = int((1-self.bottom_k_percentile)*len(sum_abs_corrs))
        self.current_features = list(sum_abs_corrs.index[min_index_to_select:])
        self.selected_features += self.current_features
