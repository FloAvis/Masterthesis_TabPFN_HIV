"""Implementation of the Binary relevance Multilable prediction algorithm using TabPFN and the HIV drug resistance dataset
as an example"""

# Setup Imports
import pandas as pd
import numpy as np
import time

import sys
import os

import utils

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate

from scipy.stats import pearsonr

from sklearn.preprocessing import OneHotEncoder

# Baseline Imports

from tabpfn import TabPFNClassifier


class BinaryRelevance():

    def __init__(self, estimator, **tabpfn_params):
        # Store parameters
        self.tabpfn_params = tabpfn_params
        self.estimator = estimator

        # Initialize the underlying classifier with given parameters
        #self.clf = self.estimator(**tabpfn_params)



    def fit(self, X, Y, sample_weight=None, **fit_params):

        self.estimators_ = []

        for i in range(Y.shape[1]):
            self.estimators_.append(self.estimator(self.tabpfn_params).fit(X, Y[:, i]))

        return self


    def predict(self, X):

        y = []

        for e in self.estimators_:
            y.append(e.predict(X))


        return np.asarray(y).T

    def predict_proba(self, X):

        results = [estimator.predict_proba(X) for estimator in self.estimators_]
        return results



def main():
    files = [r"../data/PI_DataSet.txt", r"../data/INI_DataSet.txt", r"../data/NRTI_DataSet.txt",
             r"../data/NNRTI_DataSet.txt"]

    for file in files:

        # Reading in and processing high quality File
        df = pd.read_csv(file, sep='\t')

        # removing index and summary column
        df = df.iloc[:, 1:-1]

        # list of current drugs of the dataset
        drugs = [drug for drug in list(df.columns) if not drug.startswith("P")]

        # Filtering out drugs with less than 10 labels present
        unusable_drugs = [drug for drug in drugs if df[drug].count() <= 10]

        if len(unusable_drugs) > 0:
            df.drop(columns=unusable_drugs, inplace=True)

            drugs = [drug for drug in drugs if drug not in unusable_drugs]

        # dropping rows with na labels
        df.dropna(subset=drugs, inplace=True)

        X = df.drop(drugs, axis=1)

        Y = utils.get_classes(df, drugs, mode="binary")

        clf = TabPFNClassifier()

        multi_target_pfn = BinaryRelevance(clf)

        use_kfold = False

        folds = 5

        if use_kfold == False:

            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

            trained_model_pfn = multi_target_pfn.fit(X_train, y_train)

            y_pred = trained_model_pfn.predict(X_test)

            y_pred_df = pd.DataFrame(y_pred, columns=drugs)

            y_test_df = pd.DataFrame(y_test, columns=drugs)

            utils.save_multilabel(y_pred_df, y_test_df, label=(
                        file.split("/")[-1].split("_")[0] + "_results/" + file.split("/")[-1].split("_")[
                    0] + "_Binary_Relevance_homebrew_prediction"))

            y_pred_proba = trained_model_pfn.predict_proba(X_test)

            utils.save_multilabel_proba(y_pred_proba, y_test_df, label=(
                    file.split("/")[-1].split("_")[0] + "_results/" + file.split("/")[-1].split("_")[
                0] + "_Binary_Relevance_probabilities_homebrew_prediction"))

        else:

            kf = KFold(n_splits=folds, random_state=42, shuffle=True)

            y_pred = cross_val_predict(multi_target_pfn, X, Y, cv=kf, method="predict_proba")

            y_pred_df = pd.DataFrame(utils.calc_labels(y_pred), columns=drugs)

            kfolds = np.zeros((y_pred[0].shape[0], 1))

            k = 0

            for _, test in kf.split(X, Y):
                for i in test:
                    kfolds[i] = k
                k += 1

            # y_pred_df["kFolds"] = kfolds
            """
            y_test = np.zeros((y_pred[0].shape[0], Y.shape[1]))

            t = 0

            for _, test in kf.split(X, Y):
                for i in test:
                    # print(i)
                    for j in range(Y.shape[1]):
                        y_test[t, j] = Y.iloc[i, j]
                    t += 1

            """

            # y_test_df = pd.DataFrame(y_test, columns=drugs)

            utils.save_multilabel(y_pred_df, Y, k_folds=kfolds, label=(
                    file.split("/")[-1].split("_")[0] + "_results/" + file.split("/")[-1].split("_")[
                0] + "_Binary_Relevance_" + str(folds) + "_fold_MOC_prediction"))

            utils.save_multilabel_proba(y_pred, Y, k_folds=kfolds, label=(
                    file.split("/")[-1].split("_")[0] + "_results/" + file.split("/")[-1].split("_")[
                0] + "_Binary_Relevance_probabilities_" + str(folds) + "_fold_MOC_prediction"))


if __name__ == '__main__':
    main()