"""Implementation of the Classifier Chain Multilable prediction algorithm using TabPFN and the HIV drug resistance dataset
as an example"""

# Setup Imports
import pandas as pd
import numpy as np


import prediction_handler
import data_preprocessing
import result_handler


from sklearn.model_selection import KFold

# Baseline Imports

from tabpfn import TabPFNClassifier
from Classifiers import ClassifierChains as cc
from Classifiers import Ensemble as en


# Running the predictions
def main():

    files = [r"../data/INI_DataSet.txt", r"../data/NRTI_DataSet.txt",
             r"../data/NNRTI_DataSet.txt",r"../data/PI_DataSet.txt"]

    for file in files:

        version = "_" + file.split("/")[-1].strip(".csv") + "_wo_NaN"

        X, Y, drugs = data_preprocessing.hq_hiv_loader(file, drop_na=True)


        multi_target_pfn = cc(TabPFNClassifier, random_state=42)


        # number of members of ensembles
        n_jobs = 4


        ensemble = en(cc, random_state=42, n_jobs=n_jobs)

        #kfold splitting
        folds = 5

        kf = KFold(n_splits=folds, random_state=42, shuffle=True)


        # cross validation
        y_pred, y_true = prediction_handler.cv_predict(ensemble, X, Y, cv=kf, mode="ensemble", method="predict_proba")

        df_y_true = pd.DataFrame(y_true, columns=drugs)

        # transformation of probabilities into labels
        y_pred_labels = (y_pred[...,1] >= 0.5) * 1.0

        # getting the kfold split
        kfolds = result_handler.get_kfold(kf, X, Y)

        y_pred_proba_new = []

        for proba in y_pred:
            y_pred_proba_new.append(np.stack(proba, axis=1))

        #saving labels
        result_handler.save_ensemble(y_pred_labels, df_y_true, k_folds=kfolds.flatten(), label=(
                    file.split("/")[-1].split("_")[0] + "_results/" + file.split("/")[-1].split("_")[
                0] + "_Classifier_Chain_" + str(folds) + "_folds_ensemble_wo_NaN"))

        #saving probabilities
        result_handler.save_ensemble_proba(y_pred_proba_new, df_y_true, k_folds=kfolds.flatten(), label=(
                    file.split("/")[-1].split("_")[0] + "_results/" + file.split("/")[-1].split("_")[
                0] + "_Classifier_Chain_" + str(folds) + "_folds_ensemble_probabilities_wo_NaN"))


if __name__ == '__main__':
    main()