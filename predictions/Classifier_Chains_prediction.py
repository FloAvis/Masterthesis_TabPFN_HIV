"""Implementation of the Classifier Chain Multilable prediction algorithm using TabPFN and the HIV drug resistance dataset
as an example"""

# Setup Imports
import pandas as pd
import numpy as np
import time


import utils


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_predict

# Baseline Imports

from tabpfn import TabPFNClassifier

from Classifiers import ClassifierChains as cc




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
        #df.dropna(subset=drugs, inplace=True)

        X = df.drop(drugs, axis=1)

        Y = utils.get_classes(df, drugs, mode="binary")

        #clf = TabPFNClassifier()

        multi_target_pfn = cc(TabPFNClassifier, random_state=42)

        use_kfold = True

        folds = 5

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

            multi_target_pfn2 = cc(TabPFNClassifier, random_state=42)

            kf = KFold(n_splits=folds, random_state=42, shuffle=True)


            #y_pred = cross_val_predict(multi_target_pfn2, X, Y, cv=kf,verbose=2, method="predict_proba")

            y_pred, y_true = utils.cv_predict_proba(multi_target_pfn2, X, Y, cv=kf, method="single")

            print(y_pred.shape)

            df_y_true = pd.DataFrame(y_true, columns=drugs)

            y_pred_new = (y_pred[...,1] >= 0.5) * 1.0

            # changed the saving mechanism of classifier chain, new way is better but I don't wanna change my system so gotta convert back again
            #y_pred_new = np.stack(y_pred_new, axis=1)

            print(y_pred_new.shape)

            y_pred_df = pd.DataFrame(y_pred_new, columns=drugs)

            kfolds = np.zeros((y_pred_new.shape[0], 1))

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

            print(y_pred_df)
            print(y_pred)

            utils.save_multilabel(y_pred_df, df_y_true, k_folds=kfolds, label=(
                    file.split("/")[-1].split("_")[0] + "_results/" + file.split("/")[-1].split("_")[
                0] + "_Classifier_Chain_" + str(folds) + "_fold_homebrew_prediction_new_save"))

            utils.save_multilabel_proba(np.stack(y_pred, axis=1), df_y_true, k_folds=kfolds, label=(
                    file.split("/")[-1].split("_")[0] + "_results/" + file.split("/")[-1].split("_")[
                0] + "_Classifier_Chain_probabilities_" + str(folds) + "_fold_homebrew_prediction_new_save"))


if __name__ == '__main__':
    main()