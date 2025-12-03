"""Implementation of the Classifier Chain Multilable prediction algorithm using TabPFN and the HIV drug resistance dataset
as an example"""

# Setup Imports
import pandas as pd
import numpy as np


import prediction_handler
import data_preprocessing


from sklearn.model_selection import KFold


# Baseline Imports

from tabpfn import TabPFNClassifier

import result_handler
from Classifiers import ClassifierChains as cc


def main():

    # setting the datasets
    files = [r"../data/PI_DataSet.txt", r"../data/INI_DataSet.txt", r"../data/NRTI_DataSet.txt",
             r"../data/NNRTI_DataSet.txt"]


    folds = 5

    for file in files:

        # naming scheme for data
        version = "_" + file.split("/")[-1].strip("_DataSet.txt") + "_wo_NaN"

        # loading the dataset
        X, Y, drugs = data_preprocessing.hq_hiv_loader(file, drop_na=False)


        #command for setting up the classifier chain
        multi_target_pfn = cc(TabPFNClassifier, random_state=42)


        #kfold splitting
        kf = KFold(n_splits=folds, random_state=42, shuffle=True)

        # cross validation
        y_pred, y_true = prediction_handler.cv_predict(multi_target_pfn, X, Y, cv=kf, mode="single", method="predict_proba")

        df_y_true = pd.DataFrame(y_true, columns=drugs)


        # transformation of probabilities into labels
        y_pred_new = (y_pred[...,1] >= 0.5) * 1.0

        y_pred_df = pd.DataFrame(y_pred_new, columns=drugs)


        # getting the kfold split
        kfolds = result_handler.get_kfold(kf, X, Y)

        # saving labels
        result_handler.save_multilabel(y_pred_df, df_y_true, k_folds=kfolds, label=(
                file.split("/")[-1].split("_")[0] + "_results/" + file.split("/")[-1].split("_")[
            0] + "_Classifier_Chain_" + str(folds) + "_fold" + version))

        # saving probabilities
        result_handler.save_multilabel_proba(np.stack(y_pred, axis=1), df_y_true, k_folds=kfolds, label=(
                file.split("/")[-1].split("_")[0] + "_results/" + file.split("/")[-1].split("_")[
            0] + "_Classifier_Chain_probabilities_" + str(folds) + "_fold" + version))


if __name__ == '__main__':
    main()