import glob
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.optimize import fmin_powell
from sklearn.metrics import classification_report, confusion_matrix

from kappa import quadratic_weighted_kappa
from utils import grade_on_a_curve, find_ensemble_weights


class Ensemble(object):

    def __init__(self,
                 optfun=fmin_powell,
                 pred_paths=None,
                 cutoff=.6,
                 grade_predictions_on_a_curve=True,
                 verbose=True):
        self.optfun = optfun
        self.pred_paths = pred_paths
        self.cutoff = cutoff
        self.grade_predictions_on_a_curve = grade_predictions_on_a_curve
        self.verbose = verbose
        self.y_train = pd.read_csv("data/train.csv")["median_relevance"]
        self.predictions_dfs, self.qwks = self._load_predictions()
        self.classes = [1, 2, 3, 4]
        self.optw_ = None
        self.ensembled_oof_predictions_ = None
        self.ensembled_oof_qwk_ = None
        self.classification_report_ = None
        self.confusion_matrix_ = None

    def _load_predictions(self):
        dfs, model_names, qwks = [], [], []
        preds_location = self.pred_paths if self.pred_paths is not None else glob.glob("predictions/*.pkl")

        for pred_fn in preds_location:
            try:
                pred_obj = pickle.load(open(pred_fn, "rb"))
                if self.verbose:
                    print pred_fn, pred_obj["cv_test_errors"]
                if pred_obj["cv_test_errors"] > self.cutoff:
                    qwks.append(pred_obj["cv_test_errors"])
                    dfs.append(pred_obj["predictions"]["prediction"])
                    model_names.append(pred_fn)
            except EOFError:  # 0 size file
                pass

        assert all(df.shape == dfs[0].shape for df in dfs)

        dfs = pd.concat(dfs, axis=1)
        dfs.columns = model_names
        if self.verbose:
            print '\n' + str(dfs.shape[1]) + ' predictions loaded.'

        return dfs, qwks

    def _ensemble_predictions(self, list_of_predictions, ensemble_weights):
        ensembled_preds = np.zeros(list_of_predictions[0].shape[0])
        for preds, w in zip(list_of_predictions, ensemble_weights):
            ensembled_preds += w*preds

        if self.grade_predictions_on_a_curve:
            ensembled_preds = grade_on_a_curve(ensembled_preds, self.y_train)
        return ensembled_preds

    def ensemble_predictions(self):
        list_of_predictions = [np.array(self.predictions_dfs.ix[:(len(self.y_train) - 1), col]) for col in self.predictions_dfs.columns]
        self.optw_ = find_ensemble_weights(
            optfun=self.optfun,
            predictions=list_of_predictions,
            y_true=self.y_train,
            w_init=np.array(self.qwks) ** 0.5,
            verbose=self.verbose
        )

        self.ensembled_oof_predictions_ = self._ensemble_predictions(
            list_of_predictions=list_of_predictions,
            ensemble_weights=self.optw_
        )

        self.ensembled_oof_qwk_ = quadratic_weighted_kappa(self.ensembled_oof_predictions_, self.y_train)

    def inspect_results(self):
        if self.ensembled_oof_predictions_ is None:
            sys.exit('You must call `ensemble_predictions()` before inspecting results')

        self.classification_report_ = classification_report(self.y_train, self.ensembled_oof_predictions_)
        conf_mat = confusion_matrix(self.y_train, self.ensembled_oof_predictions_)
        self.confusion_matrix_ = pd.DataFrame(conf_mat, columns=self.classes, index=self.classes)

        print self.classification_report_
        print '----------------------------------------------------\n'
        print self.confusion_matrix_

    def create_submission(self, sub_path):
        list_of_predictions_ld = [np.array(self.predictions_dfs.ix[len(self.y_train):, col]) for col in self.predictions_dfs.columns]

        final_prediction_ld_ = self._ensemble_predictions(
            list_of_predictions=list_of_predictions_ld,
            ensemble_weights=self.optw_,
        )

        sub_path = sub_path.replace('.csv', '')
        sub_path += '_CV_' + str(np.round(self.ensembled_oof_qwk_, 6)) + '_.csv'
        submission = pd.read_csv("data/sampleSubmission.csv")
        submission["prediction"] = final_prediction_ld_
        submission.to_csv(sub_path, index=False)

if __name__ == "__main__":
    ensemble = Ensemble(optfun=fmin_powell)
    ensemble.ensemble_predictions()
    ensemble.create_submission("submission.csv")
