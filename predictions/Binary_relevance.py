"""Implementation of the Binary relevance Multilable prediction algorithm using TabPFN and the HIV drug resistance dataset
as an example"""

# Setup Imports
import pandas as pd
import numpy as np

from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import KFold

import data_preprocessing
import prediction_handler
import result_handler

# Baseline Imports

from Classifiers import BinaryRelevance as br
from tabpfn import TabPFNClassifier
from tabicl import TabICLClassifier


# Running the predictions
def main():


    files = [r"../data/PI_DataSet.txt", r"../data/INI_DataSet.txt", r"../data/NRTI_DataSet.txt", r"../data/NNRTI_DataSet.txt"]


    for file in files:

        version = "_" + file.split("/")[-1].strip(".csv") +"_wo_NaN"

        X, Y, drugs = data_preprocessing.hq_hiv_loader(file, drop_na=False)


        # Command for the custom binary relevance for the inclusion of NA values
        multi_target_pfn = br(TabPFNClassifier, random_state=42)

        # Public library command for binary relevance
        #multi_target_pfn = MultiOutputClassifier(TabICLClassifier(random_state=42), n_jobs=2)


        #kfold splitting
        folds = 5

        kf = KFold(n_splits=folds, random_state=42, shuffle=True)

        # cross validation
        y_pred, y_true = prediction_handler.cv_predict(multi_target_pfn, X, Y, cv=kf, mode="single", method="predict_proba")

        df_y_true = pd.DataFrame(y_true, columns=drugs)

        # transformation of probabilities into labels
        y_pred_new = (y_pred[..., 1] >= 0.5) * 1.0


        y_pred_df = pd.DataFrame(y_pred_new, columns=drugs)

        # getting the kfold split
        kfolds = result_handler.get_kfold(kf, X, Y)

        #saving labels
        result_handler.save_multilabel(y_pred_df, df_y_true, k_folds=kfolds, label=(
                    file.split("/")[-1].split("_")[0] + "_results/" + file.split("/")[-1].split("_")[
                0] + "_Binary_Relevance_"+ str(folds) + "_fold" + version))


        #saving probabilities
        result_handler.save_multilabel_proba(np.stack(y_pred, axis=1), df_y_true, k_folds=kfolds, label=(
                file.split("/")[-1].split("_")[0] + "_results/" + file.split("/")[-1].split("_")[
            0] + "_Binary_Relevance_probabilities_"+ str(folds) + "_fold" + version))



if __name__ == '__main__':
    main()