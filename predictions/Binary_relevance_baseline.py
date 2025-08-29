"""Baseline Binary relevance models for comparison"""

# Setup Imports
import pandas as pd
import numpy as np
import time

import sys
import os


import utils
import torch

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

# Scikit-Learn: Models
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
    cross_val_predict
)
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from xgboost import XGBClassifier, XGBRegressor

# Other ML Models
from catboost import CatBoostClassifier, CatBoostRegressor

# This transformer will be used to handle categorical features for the baseline models
column_transformer = make_column_transformer(
    (
        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        make_column_selector(dtype_include=["object", "category"]),
    ),
    remainder="passthrough",
)



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

        # Encode target labels to classes for baselines
        #le = LabelEncoder()
        #y = le.fit_transform(Y)


        # Define models
        models = [
            (
                "RandomForest",
                make_pipeline(
                    column_transformer,  # string data needs to be encoded for model
                    RandomForestClassifier(random_state=42),
                ),
            ),
            (
                "XGBoost",
                make_pipeline(
                    column_transformer,  # string data needs to be encoded for model
                    XGBClassifier(random_state=42),
                ),
            ),
            (
                "CatBoost",
                make_pipeline(
                    column_transformer,  # string data needs to be encoded for model
                    CatBoostClassifier(random_state=42, verbose=0),
                ),
            ),
        ]

        use_kfold = True

        folds = 5

        kf = KFold(n_splits=folds, random_state=42, shuffle=True)



        for name, model in models:

            multi_target = MultiOutputClassifier(model, n_jobs=2)

            if use_kfold == False:

                X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)


                trained_model = multi_target.fit(X_train, y_train)
                y_pred =  trained_model.predict(X_test)

                y_pred_df = pd.DataFrame(y_pred, columns=drugs)

                y_test_df = pd.DataFrame(y_test, columns=drugs)

                utils.save_multilabel(y_pred_df, y_test_df, label=(
                            file.split("/")[-1].split("_")[0] + "_results/" + file.split("/")[-1].split("_")[
                        0] +"_" + name + "_Binary_Relevance_MOC_prediction"))

                try:
                    y_pred_proba = trained_model.predict_proba(X_test)

                    # y_pred_df = pd.DataFrame(y_pred_proba, columns=drugs)

                    # y_test_df = pd.DataFrame(y_test, columns=drugs)

                    utils.save_multilabel_proba(y_pred_proba, y_test_df, label=(
                            file.split("/")[-1].split("_")[0] + "_results/" + file.split("/")[-1].split("_")[
                        0] + "_" + name + "_Binary_Relevance_probabilities_MOC_prediction"))
                except:
                    print("Model " + name + " cannot return probabilities")

            else:


                y_pred = cross_val_predict(multi_target, X, Y, cv=kf, method="predict_proba")

                y_pred_df = pd.DataFrame(utils.calc_labels(y_pred), columns=drugs)

                kfolds = np.zeros((y_pred[0].shape[0], 1))

                k = 0

                for _, test in kf.split(X, Y):
                    for i in test:
                        kfolds[i] = k
                    k += 1

                #y_pred_df["kFolds"] = kfolds

                y_test = np.zeros((y_pred[0].shape[0], Y.shape[1]))

                t = 0

                for _, test in kf.split(X, Y):
                    for i in test:
                        # print(i)
                        for j in range(Y.shape[1]):
                            y_test[t, j] = Y.iloc[i, j]
                        t += 1

                y_test_df = pd.DataFrame(y_test, columns=drugs)

                utils.save_multilabel(y_pred_df, y_test_df, k_folds=kfolds, label=(
                        file.split("/")[-1].split("_")[0] + "_results/" + file.split("/")[-1].split("_")[
                    0] + "_" + name + "_Binary_Relevance_"+ str(folds) + "_fold_MOC_prediction"))

                utils.save_multilabel_proba(y_pred, y_test_df, k_folds=kfolds, label=(
                        file.split("/")[-1].split("_")[0] + "_results/" + file.split("/")[-1].split("_")[
                    0] + "_" + name + "_Binary_Relevance_probabilities_"+ str(folds) + "_fold_MOC_prediction"))



if __name__ == '__main__':
    main()