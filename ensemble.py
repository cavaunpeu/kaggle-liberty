import glob
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.optimize import fmin_powell

from utils import PREDICTION_PATH, find_ensemble_weights


class Ensemble(object):

    def __init__(self, opt_func=fmin_powell, cutoff=.38, verbose=True):
        self.opt_func = opt_func
        self.path = PREDICTION_PATH
        self.cutoff = cutoff
        self.verbose = verbose
        self.oof_predictions, self.lb_predictions, self.oof_ginis = self._load_predictions()
        self.opt_weights = None
        self.ensembled_oof_predictions_ = None

    def _load_predictions(self):
        oof_predictions, lb_predictions, names, oof_ginis = [], [], [], []

        for f in glob.glob(self.path + '/*'):
            try:
                pred = pickle.load(open(f, 'rb'))
                oof_gini = pred['normalized_gini']
                pred_name = str(f.split('/')[-1].split('.')[0])
                if oof_gini > self.cutoff:
                    if self.verbose:
                        print '{} | CV: {}'.format(pred_name, oof_gini)
                    oof_predictions.append(pred['oof_predictions'])
                    lb_predictions.append(pred['lb_predictions'])
                    names.append(pred_name)
                    oof_ginis.append(pred['normalized_gini'])
            except EOFError:  # 0 size file
                continue

        oof_predictions = pd.DataFrame(oof_predictions).T
        oof_predictions.columns = names
        lb_predictions = pd.DataFrame(lb_predictions).T
        lb_predictions.columns = names

        if self.verbose:
            print '\n{} predictions loaded.'.format(str(len(names)))

        return oof_predictions, lb_predictions, oof_ginis


    def _ensemble_predictions(self, list_of_predictions, ensemble_weights):
        ensembled_preds = np.zeros(list_of_predictions[0].shape[0])
        for preds, w in zip(list_of_predictions, ensemble_weights):
            ensembled_preds += w*preds

        if self.grade_predictions_on_a_curve:
            ensembled_preds = grade_on_a_curve(ensembled_preds, self.y_train)
        return ensembled_preds

#     def ensemble_predictions(self):
#         list_of_predictions = [np.array(self.predictions_dfs.ix[:(len(self.y_train) - 1), col]) for col in self.predictions_dfs.columns]
#         self.optw_ = find_ensemble_weights(
#             opt_func=self.opt_func,
#             predictions=list_of_predictions,
#             y_true=self.y_train,
#             w_init=np.array(self.qwks) ** 0.5,
#             verbose=self.verbose
#         )

#         self.ensembled_oof_predictions_ = self._ensemble_predictions(
#             list_of_predictions=list_of_predictions,
#             ensemble_weights=self.optw_
#         )

#         self.ensembled_oof_qwk_ = quadratic_weighted_kappa(self.ensembled_oof_predictions_, self.y_train)

#     def create_submission(self, sub_path):
#         list_of_predictions_ld = [np.array(self.predictions_dfs.ix[len(self.y_train):, col]) for col in self.predictions_dfs.columns]

#         final_prediction_ld_ = self._ensemble_predictions(
#             list_of_predictions=list_of_predictions_ld,
#             ensemble_weights=self.optw_,
#         )

#         sub_path = sub_path.replace('.csv', '')
#         sub_path += '_CV_' + str(np.round(self.ensembled_oof_qwk_, 6)) + '_.csv'
#         submission = pd.read_csv("data/sampleSubmission.csv")
#         submission["prediction"] = final_prediction_ld_
#         submission.to_csv(sub_path, index=False)

# if __name__ == "__main__":
#     ensemble = Ensemble(opt_func=fmin_powell)
#     ensemble.ensemble_predictions()
#     ensemble.create_submission("submission.csv")
