"""Implementation of the Classifier Chain Multilable prediction algorithm using TabPFN and the HIV drug resistance dataset
as an example"""

# Setup Imports
import pandas as pd
import numpy as np
import time


import prediction_handler
import data_preprocessing

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_predict

# Baseline Imports

from tabpfn import TabPFNClassifier

import result_handler
from Classifiers import ClassifierChains as cc

#from tabicl import TabICLClassifier


def main():

    #files = [r"../data/PI_DataSet.txt", r"../data/INI_DataSet.txt", r"../data/NRTI_DataSet.txt",
    #         r"../data/NNRTI_DataSet.txt"]

    files = [r"../data/Other_datasets/scene.csv"]

    feature_prefix = "F"
    label_prefix = "T"

    version = "_scene_data_wo_NaN"

    use_kfold = True

    folds = 5

    for file in files:

        #X, Y, drugs = data_preprocessing.hq_hiv_loader(file, drop_na=True)

        df = pd.read_csv(file, true_values=["b'1'"], false_values=["b'0'"], dtype=float)

        X = df.filter(regex=feature_prefix)
        Y = df.filter(regex=label_prefix)

        # print(X)
        # print(Y)

        drugs = list(Y.columns.values)

        multi_target_pfn = cc(TabPFNClassifier, random_state=42)

        if not use_kfold:

            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

            multi_target_pfn.fit(X_train, y_train)

            y_pred = multi_target_pfn.predict(X_test)

            y_pred_df = pd.DataFrame(y_pred, columns=drugs)

            y_test_df = pd.DataFrame(y_test, columns=drugs)

            utils.save_multilabel(y_pred_df, y_test_df, label=(
                        file.split("/")[-1].split("_")[0] + "_results/" + file.split("/")[-1].split("_")[
                    0] + "_Classifier_Chain_homebrew_prediction"))

            y_pred_proba = multi_target_pfn.predict_proba(X_test)


            #changed the saving mechanism of classifier chain, new way is better but I don't wanna change my system so gotta convert back again
            y_pred_proba_new = np.stack(y_pred_proba, axis=1)

            utils.save_multilabel_proba(y_pred_proba_new, y_test_df, label=(
                    file.split("/")[-1].split("_")[0] + "_results/" + file.split("/")[-1].split("_")[
                0] + "_Classifier_Chain_probabilities_homebrew_prediction"))

        else:

            kf = KFold(n_splits=folds, random_state=42, shuffle=True)


            #y_pred = cross_val_predict(multi_target_pfn2, X, Y, cv=kf,verbose=2, method="predict_proba")

            y_pred, y_true = prediction_handler.cv_predict_proba(multi_target_pfn, X, Y, cv=kf, method="single")

            print(y_pred.shape)

            df_y_true = pd.DataFrame(y_true, columns=drugs)

            y_pred_new = (y_pred[...,1] >= 0.5) * 1.0

            print(y_pred_new.shape)

            y_pred_df = pd.DataFrame(y_pred_new, columns=drugs)

            kfolds = result_handler.get_kfold(kf, X, Y)

            # y_test_df = pd.DataFrame(y_test, columns=drugs)

            print(y_pred_df)
            print(y_pred)

            result_handler.save_multilabel(y_pred_df, df_y_true, k_folds=kfolds, label=(
                    file.split("/")[-1].split("_")[0] + "_results/" + file.split("/")[-1].split("_")[
                0] + "_Classifier_Chain_" + str(folds) + "_fold" + version))

            result_handler.save_multilabel_proba(np.stack(y_pred, axis=1), df_y_true, k_folds=kfolds, label=(
                    file.split("/")[-1].split("_")[0] + "_results/" + file.split("/")[-1].split("_")[
                0] + "_Classifier_Chain_probabilities_" + str(folds) + "_fold" + version))


if __name__ == '__main__':
    main()