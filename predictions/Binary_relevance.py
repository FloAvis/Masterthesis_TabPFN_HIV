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

from scipy.stats import pearsonr

from sklearn.preprocessing import OneHotEncoder

# Baseline Imports

from tabpfn import TabPFNClassifier


class BinaryRelevanceTabPFN():

    def predict(x, y):
        """
        #result dataframe
        results = pd.DataFrame(columns=["Drug",
                                        "Samples",
                                        "Accuracy",
                                        "Pearson",
                                        "F1",
                                        "AUC PRC",
                                        "AUC ROC",
                                        "Time"])



        for drug in drugs:
            print(input_file.split("/")[1].split("_")[0] + ": " + drug)
            tmp_drugs = drugs.copy().remove(drug)

            #getting labels of only needed drug
            #dataframe = df.drop(tmp_drugs, axis=1)

            dataframe = df.dropna(subset=[drug])

            #If no thresholds for drug available no prediction possible
            if drug not in utils.THRESHOLD_INDICES:
                results = pd.concat([pd.DataFrame([[drug, dataframe.shape[0], None, None, None,
                                                    None, None, None]], columns=results.columns),
                                     results], ignore_index=True)
                continue


            # encoding the levels of susceptibility as 0 for susceptible, 1 as partly resistant and 2 as completly resistant
            y = utils.get_classes(dataframe, drug, mode="multiclass")


            X = dataframe.drop([drug], axis=1)



            #X_trafo = enc.transform(X).toarray()


            #----------------------------------------------------------------------------------------------------------------
            #Training


            #getting train test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

            #Timing TabPFN
            start_time = time.time()

            # Train and evaluate TabPFN
            y_pred = TabPFNClassifier(random_state=42, ignore_pretraining_limits=True).fit(X_train, y_train).predict_proba(X_test)

            taken_time = time.time() - start_time

            y_pred_class = np.argmax(y_pred, axis=1)

            #--------------------------------------------------------------------------------------------------------------------
            # Evaluation metrics

            #Accuracy:
            scores = {"Accuracy": accuracy_score(y_test, y_pred_class)}

            #Person coefficient:
            scores.update({"Pearson": pearsonr(y_test, y_pred_class)[0]})

            #F1 score:
            scores.update({"F1": f1_score(y_test, y_pred_class, average="micro")})

            # Calculate PRC AUC
            scores.update({"AUC PRC" : utils.prc_auc_score(y_test, y_pred, multiclass="ovr")})
            #print(f"TabPFN PRC AUC: {score_prc:.4f}")

            # Calculate ROC AUC (handles both binary and multiclass)
            scores.update({ "AUC ROC": roc_auc_score(y_test, y_pred if len(np.unique(y)) > 2 else y_pred[:, 1], multi_class='ovr')})
            #print(f"TabPFN ROC AUC: {score_roc:.4f}")


            #saving the resulting statistics
            results = pd.concat([pd.DataFrame([[
                                                drug,
                                                X.shape[0],
                                                scores["Accuracy"],
                                                scores["Pearson"],
                                                scores["F1"],
                                                scores["AUC PRC"],
                                                scores["AUC ROC"],
                                                taken_time
                                            ]], columns=results.columns), results], ignore_index=True)


            """




        pass

        #saving results:
        #utils.save_results(y_pred, y_test, label= (input_file.split("/")[1].split("_")[0] + "_results/" + drug + "_results/" + "Multilabel_prediction"))


        #return results

def main():


    files = [r"../data/PI_DataSet.txt", r"../data/INI_DataSet.txt", r"../data/NRTI_DataSet.txt", r"../data/NNRTI_DataSet.txt"]

    for file in files:
        #running_models(file, "../output/" + (file.split("/")[-1].strip(".txt") + "_multilabel_results.csv"))

        # Reading in and processing high quality File
        df = pd.read_csv(file, sep='\t')

        # removing index and summary column
        df = df.iloc[:, 1:-1]

        # list of current drugs of the dataset
        drugs = [drug for drug in list(df.columns) if not drug.startswith("P")]

        #Filtering out drugs with less than 10 labels present
        unusable_drugs = [drug for drug in drugs if df[drug].count() <= 10]

        if len(unusable_drugs) > 0:
            df.drop(columns=unusable_drugs, inplace=True)

            drugs = [drug for drug in drugs if drug not in unusable_drugs]

        df.dropna(subset=drugs, inplace=True)


        # creating the one hot encoding for the features
        #enc = OneHotEncoder(handle_unknown='error')

        #enc.fit(df.loc[:, [drug for drug in list(df.columns) if drug.startswith("P")]])



        X = df.drop(drugs, axis=1)




        Y = utils.get_classes(df, drugs, mode="binary")

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

        clf = TabPFNClassifier()


        multi_target_pfn = MultiOutputClassifier(clf, n_jobs=2)

        """
        y_pred = multi_target_pfn.fit(X_train, y_train).predict(X_test)



        #BR = BinaryRelevanceTabPFN()


        #results = BR.predict(X, Y)

        y_pred_df = pd.DataFrame(y_pred, columns=drugs)

        y_test_df = pd.DataFrame(y_test, columns=drugs)


        utils.save_multilabel(y_pred_df, y_test_df, label= (file.split("/")[-1].split("_")[0] + "_results/" + file.split("/")[-1].split("_")[0] + "_Binary_Relevance_MOC_prediction"))
        """

        y_pred_proba = multi_target_pfn.fit(X_train, y_train).predict_proba(X_test)

        #y_pred_df = pd.DataFrame(y_pred_proba, columns=drugs)

        y_test_df = pd.DataFrame(y_test, columns=drugs)

        utils.save_multilabel_proba(y_pred_proba, y_test_df, label=(
                    file.split("/")[-1].split("_")[0] + "_results/" + file.split("/")[-1].split("_")[
                0] + "_Binary_Relevance_probabilities_MOC_prediction"))



if __name__ == '__main__':
    main()