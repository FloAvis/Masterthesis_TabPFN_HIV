"""This script calculates different evaluation metrics of base prediction settings of TabPFN
for the comparison with the multilabel and other approaches"""

# Setup Imports
import pandas as pd
import numpy as np
import time

import sys
import os

#sys.path.append(os.path.abspath('..'))

import prediction_handler

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from scipy.stats import pearsonr

from sklearn.preprocessing import OneHotEncoder

# Baseline Imports

#from tabpfn import TabPFNClassifier

from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier

from autogluon.features.generators import AutoMLPipelineFeatureGenerator

# table for the encoding of the resistance testing into three classes: "susceptible", "intermediate-level resistant", "high-level resistant" with lower and upper thresholds

def running_models(input_file, output_file):


    # Reading in and processing high quality File
    df = pd.read_csv(input_file, sep='\t')

    #removing index and summary column
    df = df.iloc[:,1:-1]


    #list of current drugs of the dataset
    drugs = [drug for drug in list(df.columns) if not drug.startswith("P") ]

    #creating the one hot encoding for the features
    enc = OneHotEncoder(handle_unknown='error')

    enc.fit(df.loc[:,[drug for drug in list(df.columns) if drug.startswith("P")]])

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

        tmp_drugs = drugs.copy()
        tmp_drugs.remove(drug)

        #print(drugs)

        print(tmp_drugs)

        # getting labels of only needed drug
        dataframe = df.drop(tmp_drugs, axis=1)

        dataframe = dataframe.dropna()

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

        #print(X_trafo)

        print(X)

        #getting train test split
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        feature_generator = AutoMLPipelineFeatureGenerator()
        feature_generator.fit(X_train_raw)

        # Transform the datasets
        X_train = feature_generator.transform(X_train_raw)
        X_test = feature_generator.transform(X_test_raw)

        print(X_test.shape)
        print(X_train.shape)

        #Timing TabPFN
        start_time = time.time()

        # Train and evaluate TabPFN
        y_pred = AutoTabPFNClassifier(random_state=42, ignore_pretraining_limits=True, max_time=120).fit(X_train, y_train).predict_proba(X_test)

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



        #saving results:
        utils.save_results(y_pred, y_test, label= (input_file.split("/")[-1].split("_")[0] + "_results/" + drug + "_results/" + "AutoTabPFN_results"))


    results.to_csv(output_file)


def main():

    files = [r"../data/PI_DataSet.txt", r"../data/INI_DataSet.txt", r"../data/NRTI_DataSet.txt", r"../data/NNRTI_DataSet.txt"]


    for file in files:
        running_models(file, "../output/" + (file.split("/")[-1].strip(".txt") + "AutoTabPFN_results.csv"))

    """
    file = r"../data/INI_DataSet.txt"

    running_models(file, "../output/" + (file.split("/")[-1].strip(".txt") + "AutoTabPFN_results.csv"))
    """
if __name__ == '__main__':
    main()