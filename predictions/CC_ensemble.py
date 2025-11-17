"""Implementation of the Classifier Chain Multilable prediction algorithm using TabPFN and the HIV drug resistance dataset
as an example"""

# Setup Imports
import pandas as pd
import numpy as np
import time


import prediction_handler
import data_preprocessing
import result_handler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_predict

# Baseline Imports

from tabpfn import TabPFNClassifier

from Classifiers import ClassifierChains as cc

from Classifiers import Ensemble as en

from sklearn.metrics import jaccard_score

def main():
    files = [r"../data/NRTI_DataSet.txt",
             r"../data/NNRTI_DataSet.txt",r"../data/PI_DataSet.txt"]

    for file in files:

        X, Y, drugs = data_preprocessing.hq_hiv_loader(file, drop_na=True)

        #clf = TabPFNClassifier()

        multi_target_pfn = cc(TabPFNClassifier, random_state=42)

        use_kfold = True

        folds = 5

        n_jobs = 4

        #X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)



        ensemble = en(cc, random_state=42, n_jobs=n_jobs)

        if not use_kfold:

            ensemble.fit(X=X_train, Y=y_train)

            y_pred = ensemble.predict(X_test)

            y_pred_proba = ensemble.predict_proba(X_test)


            y_test_df = pd.DataFrame(y_test, columns=drugs)

            y_pred_proba_new = []

            for proba in y_pred_proba:
                y_pred_proba_new.append( np.stack(proba, axis=1))

            utils.save_ensemble(y_pred, y_test_df, label=(
                        file.split("/")[-1].split("_")[0] + "_results/" + file.split("/")[-1].split("_")[
                    0] + "_Classifier_Chain_ensemble"))


            utils.save_ensemble_proba(y_pred_proba_new, y_test_df, label=(
                    file.split("/")[-1].split("_")[0] + "_results/" + file.split("/")[-1].split("_")[
                0] + "_Classifier_Chain_ensemble_probabilities"))
        else:

            kf = KFold(n_splits=folds, random_state=42, shuffle=True)

            y_pred, y_true = prediction_handler.cv_predict(ensemble, X, Y, cv=kf, mode="ensemble", method="predict_proba")

            df_y_true = pd.DataFrame(y_true, columns=drugs)

            y_pred_labels = (y_pred[...,1] >= 0.5) * 1.0

            kfolds = result_handler.get_kfold(kf, X, Y)

            y_pred_proba_new = []

            for proba in y_pred:
                y_pred_proba_new.append(np.stack(proba, axis=1))


            # y_test_df = pd.DataFrame(y_test, columns=drugs)

            result_handler.save_ensemble(y_pred_labels, df_y_true, k_folds=kfolds.flatten(), label=(
                        file.split("/")[-1].split("_")[0] + "_results/" + file.split("/")[-1].split("_")[
                    0] + "_Classifier_Chain_" + str(folds) + "_folds_ensemble_wo_NaN"))

            result_handler.save_ensemble_proba(y_pred_proba_new, df_y_true, k_folds=kfolds.flatten(), label=(
                        file.split("/")[-1].split("_")[0] + "_results/" + file.split("/")[-1].split("_")[
                    0] + "_Classifier_Chain_" + str(folds) + "_folds_ensemble_probabilities_wo_NaN"))


if __name__ == '__main__':
    main()