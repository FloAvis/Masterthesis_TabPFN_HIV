"""Implementation"""

# Setup Imports
import pandas as pd
import numpy as np

import prediction_handler
import result_handler
import data_preprocessing


from sklearn.model_selection import KFold
from sklearn.multioutput import ClassifierChain as skl_cc
from sklearn.multioutput import MultiOutputClassifier

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


from sklearn.preprocessing import OneHotEncoder


from skmultilearn.ensemble import RakelO, RakelD

import scipy


def main():

    # setting the datasets
    files = [r"../data/PI_DataSet.txt", r"../data/INI_DataSet.txt", r"../data/NRTI_DataSet.txt",r"../data/NNRTI_DataSet.txt"]


    for file in files:

        # loading the dataset
        X, Y, drugs = data_preprocessing.hq_hiv_loader(file, drop_na=False)


        #encoding the dataset
        enc = OneHotEncoder(handle_unknown='error')

        enc.fit(X)
        X_trafo = enc.transform(X).toarray()

        # kfold splitting
        folds = 5

        kf = KFold(n_splits=folds, random_state=42, shuffle=True)

        forest = RandomForestClassifier(random_state=42)
        xgb = XGBClassifier(random_state=42)
        lr = LogisticRegression()

        models = [
            ("BR_LR", MultiOutputClassifier(lr, n_jobs=2)),
            ("BR_XGB", MultiOutputClassifier(xgb, n_jobs=2)),
            ("BR_forest", MultiOutputClassifier(forest, n_jobs=2)),
            ("CC_LR", skl_cc(lr, order="random", random_state=42)),
            ("CC_xgb", skl_cc(xgb, order="random", random_state=42)),
            ("CC_forest", skl_cc(forest, order="random", random_state=42)),
            ("Rakeld_lr", RakelD(base_classifier=lr, base_classifier_require_dense=[True, True], labelset_size=2)),
            ("Rakeld_xgb", RakelD(base_classifier=xgb,base_classifier_require_dense=[True, True], labelset_size=2)),
            ("Rakeld_forest", RakelD(base_classifier=forest, base_classifier_require_dense=[True, True], labelset_size=2)),
        ]


        # predictions for all dataset
        for name, model in models:


            print(name)

            # cross validation
            y_pred, y_true = prediction_handler.cv_predict(model, X_trafo, Y, cv=kf, mode="single", method="predict_proba")

            if isinstance(y_pred, scipy.sparse._csr.csr_matrix):
                y_pred = y_pred.todense()

            df_y_true = pd.DataFrame(y_true, columns=drugs)


            # transformation of probabilities into labels
            y_pred_new = (y_pred[..., 1] >= 0.5) * 1.0


            y_pred_df = pd.DataFrame(y_pred_new, columns=drugs)


            # getting the kfold split
            kfolds = result_handler.get_kfold(kf, X_trafo, Y)


            # saving labels
            result_handler.save_multilabel(y_pred_df, df_y_true, k_folds=kfolds, label=(
                    file.split("/")[-1].split("_")[0] + "_results/benchmarkings/" + file.split("/")[-1].split("_")[
                0] + "_" + name + "_" + str(folds) + "_fold_w_NaN"))


            # saving probabilities
            result_handler.save_multilabel_proba(np.stack(y_pred, axis=1), df_y_true, k_folds=kfolds,
                                                 label=(
                    file.split("/")[-1].split("_")[0] + "_results/benchmarkings/" +
                    file.split("/")[-1].split("_")[
                        0] + "_" + name + "_" + str(folds) + "_fold_w_NaN" + "_probabilities"))

if __name__ == '__main__':
    main()