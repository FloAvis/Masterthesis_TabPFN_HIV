"""Implementation"""

# Setup Imports
import pandas as pd
import numpy as np
import time
import os

import prediction_handler
import result_handler
import data_preprocessing

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.multioutput import ClassifierChain as skl_cc
from sklearn.multioutput import MultiOutputClassifier

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from skmultilearn.problem_transform import BinaryRelevance

from sklearn.preprocessing import OneHotEncoder

from sklearn import tree
from skmultilearn.ensemble import RakelO, RakelD

import scipy


from sklearn.model_selection import cross_val_predict

# Baseline Imports

from tabpfn import TabPFNClassifier

from Classifiers import ClassifierChains as cc

from Classifiers import Ensemble as en

from sklearn.metrics import jaccard_score

def main():
    files = [r"../data/PI_DataSet.txt", r"../data/INI_DataSet.txt", r"../data/NRTI_DataSet.txt",r"../data/NNRTI_DataSet.txt"]


    for file in files:

            X, Y, drugs = data_preprocessing.hq_hiv_loader(file, drop_na=True)

            enc = OneHotEncoder(handle_unknown='error')


            enc.fit(X)
            X_trafo = enc.transform(X).toarray()

            use_kfold = True

            folds = 5


            forest = RandomForestClassifier(random_state=42)
            xgb = XGBClassifier(random_state=42)
            lr = LogisticRegression()

            models = [
                #("BR_LR", MultiOutputClassifier(lr, n_jobs=2)),
                #("BR_XGB", MultiOutputClassifier(xgb, n_jobs=2)),
                #("BR_forest", MultiOutputClassifier(forest, n_jobs=2)),
                ("CC_LR", skl_cc(lr, order="random", random_state=42)),
                ("CC_xgb", skl_cc(xgb, order="random", random_state=42)),
                ("CC_forest", skl_cc(forest, order="random", random_state=42)),
                ("Rakeld_lr", RakelD(base_classifier=lr, base_classifier_require_dense=[True, True], labelset_size=2)),
                ("Rakeld_xgb", RakelD(base_classifier=xgb,base_classifier_require_dense=[True, True], labelset_size=2)),
                ("Rakeld_forest", RakelD(base_classifier=forest, base_classifier_require_dense=[True, True], labelset_size=2)),
                #("Rakelo_lr", RakelO(base_classifier=lr, base_classifier_require_dense=[True, True], labelset_size=y_train.shape[1] // 4, model_count=6)),
                #("Rakelo_xgb", RakelO(base_classifier=xgb,base_classifier_require_dense=[True, True],labelset_size=y_train.shape[1] // 4, model_count=6)),
                #("Rakelo_forest", RakelO(base_classifier=forest, base_classifier_require_dense=[True, True], labelset_size=y_train.shape[1] // 4, model_count=6))
            ]

            #ensemble = en(cc, random_state=42, n_jobs=n_jobs)

            if not use_kfold:

                for name, model in models:
                    print()
                    model.fit(X_train, y_train)

                    y_pred = model.predict(X_test)
                    print(type(y_pred))

                    print(y_pred)

                    if isinstance(y_pred, scipy.sparse.spmatrix):
                        y_pred = y_pred.todense()

                    y_pred_df = pd.DataFrame(y_pred, columns=drugs)

                    y_test_df = pd.DataFrame(y_test, columns=drugs)

                    # print(np.array(y_pred_proba).shape)

                    utils.save_multilabel(y_pred_df, y_test_df, label=(
                            file.split("/")[-1].split("_")[0] + "_results/benchmarkings/" + file.split("/")[-1].split("_")[
                        0] + "_" + name))

                    if name.startswith("CC"):

                        y_pred_proba = model.predict_proba(X_test)
                        # y_pred_proba_new = np.stack(y_pred_proba, axis=1)
                        # print(y_pred_proba_new.shape)

                        y_pred_proba_new = pd.DataFrame(y_pred_proba, columns=drugs)

                        utils.save_multilabel(y_pred_proba_new, y_test_df, label=(
                                file.split("/")[-1].split("_")[0] + "_results/benchmarkings/" +
                                file.split("/")[-1].split("_")[
                                    0] + "_" + name + "_probabilities"))
                    elif name.startswith("Rakel"):
                        pass
                    else:
                        y_pred_proba = model.predict_proba(X_test)

                        y_pred_proba_new = y_pred_proba

                        utils.save_multilabel_proba(y_pred_proba_new, y_test_df, label=(
                                file.split("/")[-1].split("_")[0] + "_results/benchmarkings/" +
                                file.split("/")[-1].split("_")[
                                    0] + "_" + name + "_probabilities"))

            else:

                for name, model in models:

                    print(name)
                    kf = KFold(n_splits=folds, random_state=42, shuffle=True)


                    y_pred, y_true = prediction_handler.cv_predict(model, X_trafo, Y, cv=kf, mode="single", method="predict_proba")

                    if isinstance(y_pred, scipy.sparse._csr.csr_matrix):
                        y_pred = y_pred.todense()

                    df_y_true = pd.DataFrame(y_true, columns=drugs)

                    #y_pred_new = (y_pred[..., 1] >= 0.5) * 1.0

                    #print(y_pred_new.shape)

                    #y_pred_df = pd.DataFrame(y_pred_new, columns=drugs)

                    kfolds = result_handler.get_kfold(kf, X_trafo, Y)
                    # print(np.array(y_pred_proba).shape)

                    """
                    utils.save_multilabel(y_pred_df, y_test_df, k_folds=kfolds, label=(
                            file.split("/")[-1].split("_")[0] + "_results/benchmarkings/" + file.split("/")[-1].split("_")[
                        0] + "_" + name + "_" + str(folds) + "_fold"))
                    """

                    print(y_pred.shape)
                    print(np.stack(y_pred, axis=1).shape)


                    result_handler.save_multilabel_proba(np.stack(y_pred, axis=1), df_y_true, k_folds=kfolds,
                                                         label=(
                            file.split("/")[-1].split("_")[0] + "_results/benchmarkings/" +
                            file.split("/")[-1].split("_")[
                                0] + "_" + name + "_" + str(folds) + "_fold" + "_probabilities"))

if __name__ == '__main__':
    main()