"""This script performs regression for all drugs and then (if possible) assigns classes based on
the predicted values and calculates AUC ROC and PRC"""

# Setup Imports
import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.model_selection import (
    cross_val_score,
    KFold
)
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
    roc_auc_score,
    precision_recall_curve,
    auc, average_precision_score
)
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.inspection import DecisionBoundaryDisplay

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Baseline Imports
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

import torch

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier, AutoTabPFNRegressor


# table for the encoding of the resistance testing into three classes: "susceptible", "intermediate-level resistant", "high-level resistant" with lower and upper thresholds

def running_models(input_file, output_file):

    #thresholds defined by the database for the classes of "susceptible", "partly susceptible", and "resistant"
    thresholds = [
        [3, 15],  # FPV
        [3, 15],  # ATV
        [3, 15],  # IDV
        [9, 55],  # LPV
        [3, 6],  # NFV
        [3, 15],  # SQV
        [2, 8],  # TPV
        [10, 90],  # DRV
        [5, 25],  # X3TC
        [2, 6],  # ABC
        [3, 15],  # AZT
        [1.5, 3],  # D4T
        [1.5, 3],  # DDI
        [1.5, 3],  # TDF
        [3, 10],  # EFV
        [3, 10],  # NVP
        [3, 10],  # ETR
        [3, 10],  # RPV
        [2.5, 10],  # BIC
        [4, 13],  # DTG
        [2.5, 10],  # EVG - upper threshold guessed
        [1.5, 10]  # RAL - upper threshold guessed
    ]

    # Define row and column names
    index = ["FPV", "ATV", "IDV", "LPV", "NFV", "SQV", "TPV", "DRV",
             "3TC", "ABC", "AZT", "D4T", "DDI", "TDF",
             "EFV", "NVP", "ETR", "RPV", "BIC", "DTG", "EVG", "RAL"]
    columns = ["lower", "upper"]

    # Create DataFrame
    cutoff_df = pd.DataFrame(thresholds, index=index, columns=columns)

    # Reading in and processing high quality File

    df = pd.read_csv(input_file, sep='\t')
    #print(df)
    df = df.iloc[:,1:-1]
    #print(df2)

    #Checking how much data is available for each drug
    #print(df.loc[:,"FPV":"DRV"].count())

    #list of current drugs of the dataset
    drugs = [drug for drug in list(df.columns) if not drug.startswith("P") ]

    #creating the one hot encoding for the features
    enc = OneHotEncoder(handle_unknown='error')

    enc.fit(df.loc[:,[drug for drug in list(df.columns) if drug.startswith("P")]])


    #going through the drugs and splitting them to test and training depending on the drug

    results = pd.DataFrame(columns=["Drug",
                                    "RMSE",
                                    "AUC ROC MC",
                                    "AUC ROC BI",
                                    "AUC PRC BI",
                                    "R2 TabPFN",
                                    "R2 RF",
                                    "R2 XG",
                                    "R2 Cat",
                                    "Time"])

    print(input_file)

    for drug in drugs:
        #print(drug)
        tmp_drugs = drugs.copy()
        #print(tmp_drugs)
        tmp_drugs.remove(drug)
        #print(tmp_drugs)
        last_col = list(df.columns)[-1]
        dataframe = df.drop(tmp_drugs, axis=1)

        #print(dataframe.head())

        dataframe = dataframe.dropna()

        '''
        
        # encoding the levels of susceptibility as 0 for susceptible, 1 as resistant
        dataframe.loc[dataframe[drug] < cutoff_df.loc[drug, "lower"], drug + "_level_binary"] = 0
        dataframe.loc[dataframe[drug] >= cutoff_df.loc[drug, "lower"], drug + "_level_binary"] = 1

        # encoding the levels of susceptibility as 0 for susceptible, 1 as partly resistant and 2 as completly resistant
        dataframe.loc[dataframe[drug] < cutoff_df.loc[drug, "lower"], drug + "_level"] = 0
        dataframe.loc[dataframe[drug] >= cutoff_df.loc[drug, "upper"], drug + "_level"] = 2
        dataframe.loc[(dataframe[drug] >= cutoff_df.loc[drug, "lower"]) & (
                    dataframe[drug] < cutoff_df.loc[drug, "upper"]), drug + "_level"] = 1

        #print(dataframe.head())

        X, y = dataframe.drop([drug, drug + "_level"], drug + "_level_binary", axis=1), np.array(dataframe[drug])
        '''

        X, y = dataframe.drop(drug, axis=1), np.array(dataframe[drug])

        #print(y)

        X_trafo = enc.transform(X).toarray()

        #print(X_trafo.shape)

        #print(y)
        X_train, X_test, y_train, y_test = train_test_split(X_trafo, y, test_size=0.33, random_state=42)

        start_time = time.time()
        # Train and evaluate TabPFN
        y_pred = TabPFNRegressor(random_state=42, ignore_pretraining_limits=True).fit(X_train, y_train).predict(X_test)

        taken_time = time.time() - start_time
        print(drug + ":")

        # Calculate ROC AUC (handles both binary and multiclass)
        score_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"TabPFN RMSE: {score_rmse:.4f}")

        #print(y_test.shape)
        #print(y_pred.shape)
        if drug not in index: # if there are no thresholds no classification labels can be assigned to the drug
            auc_roc_mc = None
            auc_roc_bi = None
            auc_prc_bi = None
        else:
            print(y_test, y_pred)

            y_test_class = [0 if i < cutoff_df.loc[drug, "lower"] else 2 if i > cutoff_df.loc[drug, "upper"] else 1 for i in y_test]

            y_test_bi_class = [0 if i < cutoff_df.loc[drug, "lower"] else 1 for i in y_test]

            y_pred_class = [0 if i < cutoff_df.loc[drug, "lower"] else 2 if i > cutoff_df.loc[drug, "upper"] else 1 for i in
                            y_pred]

            y_pred_bi_class = [0 if i < cutoff_df.loc[drug, "lower"] else 1 for i in y_pred]

            print(y_test_class)
            print(y_pred_class)

            # Calculate ROC AUC (handles both binary and multiclass)
            auc_roc_mc = roc_auc_score(y_test_class, y_pred_class, multi_class='ovr')
            print(f"TabPFN ROC AUC Multiclass: {auc_roc_mc:.4f}")


            # Calculate ROC AUC (handles both binary and multiclass)
            auc_roc_bi = roc_auc_score(y_test_bi_class, y_pred_bi_class)
            print(f"TabPFN ROC AUC Binary: {auc_roc_bi:.4f}")


            # Calculate PRC AUC (handles currently only binary)
            tab_prec, tab_rec, thresholds = precision_recall_curve(y_test_bi_class, y_pred_bi_class)
            auc_prc_bi = auc(tab_rec, tab_prec)
            print(f"TabPFN PRC AUC: {auc_prc_bi:.4f}")



        print("-------------------------------------------------------------------------------------")

        models = [
            ("TabPFN", TabPFNRegressor(random_state=42)),
            (
                "RandomForest",
                    RandomForestRegressor(random_state=42)

            ),
            (
                "XGBoost",
                    XGBRegressor(random_state=42)

            ),
            (
                "CatBoost",
                    CatBoostRegressor(random_state=42, verbose=0)
            )
        ]

        # Calculate scores
        scoring = "r2"
        cv = KFold(n_splits=5, random_state=42, shuffle=True)
        scores = {
            name: cross_val_score(
                model, X_trafo, y, cv=cv, scoring=scoring, n_jobs=1, verbose=1
            ).mean()
            for name, model in models
        }

        #saving the resulting statistics
        results = pd.concat([pd.DataFrame([[drug,
                                            score_rmse,
                                            auc_roc_mc,
                                            auc_roc_bi,
                                            auc_prc_bi,
                                            scores['TabPFN'],
                                            scores['RandomForest'],
                                            scores['XGBoost'],
                                            scores['CatBoost'],
                                            taken_time]],
                                          columns=results.columns), results], ignore_index=True)

        for model, score in scores.items():
            print(model + ": " + str(score))

    results.to_csv(output_file)

def main():
    files = [r"data/PI_DataSet.txt", r"data/INI_DataSet.txt", r"data/NRTI_DataSet.txt", r"data/NNRTI_DataSet.txt"]

    for file in files:
        running_models(file, "output/" + (file.split("/")[-1].strip(".txt") + "_regression_results.csv"))

if __name__ == '__main__':
    main()