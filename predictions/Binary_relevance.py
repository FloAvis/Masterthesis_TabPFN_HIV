"""Implementation of the Binary relevance Multilable prediction algorithm using TabPFN and the HIV drug resistance dataset
as an example"""

# Setup Imports
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import KFold

import data_preprocessing
import prediction_handler
import result_handler

# Baseline Imports

from Classifiers import BinaryRelevance as br
from tabpfn import TabPFNClassifier
from tabicl import TabICLClassifier



def main():


    files = [r"../data/PI_DataSet.txt", r"../data/INI_DataSet.txt", r"../data/NRTI_DataSet.txt", r"../data/NNRTI_DataSet.txt"]

    for file in files:


        X, Y, drugs = data_preprocessing.hq_hiv_loader(file, drop_na=False)


        #clf = TabPFNClassifier(random_state=42)
        #clf = TabICLClassifier(random_state=42)

        multi_target_pfn = br(TabPFNClassifier, random_state=42)

        #multi_target_pfn = MultiOutputClassifier(clf, n_jobs=2)

        use_kfold = True
        folds = 5


        if use_kfold == False:

            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)


            trained_model_pfn = multi_target_pfn.fit(X_train, y_train)

            y_pred = trained_model_pfn.predict(X_test)


            y_pred_df = pd.DataFrame(y_pred, columns=drugs)

            y_test_df = pd.DataFrame(y_test, columns=drugs)


            result_handler.save_multilabel(y_pred_df, y_test_df, label= (file.split("/")[-1].split("_")[0] + "_results/" + file.split("/")[-1].split("_")[0] + "_Binary_Relevance_MOC_prediction"))


            y_pred_proba = trained_model_pfn.predict_proba(X_test)


            result_handler.save_multilabel_proba(y_pred_proba, y_test_df, label=(
                        file.split("/")[-1].split("_")[0] + "_results/" + file.split("/")[-1].split("_")[
                    0] + "_Binary_Relevance_probabilities_MOC_prediction"))

        else:

            kf = KFold(n_splits=folds, random_state=42, shuffle=True)

            y_pred, y_true = prediction_handler.cv_predict(multi_target_pfn, X, Y, cv=kf, mode="single", method="predict_proba")

            df_y_true = pd.DataFrame(y_true, columns=drugs)

            y_pred_new = (y_pred[..., 1] >= 0.5) * 1.0

            print(y_pred_new.shape)

            y_pred_df = pd.DataFrame(y_pred_new, columns=drugs)

            kfolds = result_handler.get_kfold(kf, X, Y)


            result_handler.save_multilabel(y_pred_df, df_y_true, k_folds=kfolds, label=(
                        file.split("/")[-1].split("_")[0] + "_results/" + file.split("/")[-1].split("_")[
                    0] + "_Binary_Relevance_"+ str(folds) + "_fold_homebrew_prediction_NaN_training"))



            result_handler.save_multilabel_proba(np.stack(y_pred, axis=1), df_y_true, k_folds=kfolds, label=(
                    file.split("/")[-1].split("_")[0] + "_results/" + file.split("/")[-1].split("_")[
                0] + "_Binary_Relevance_probabilities_"+ str(folds) + "_fold_homebrew_prediction_NaN_training"))



if __name__ == '__main__':
    main()