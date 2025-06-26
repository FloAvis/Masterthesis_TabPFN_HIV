# Setup Imports
import pandas as pd
import numpy as np
import time

from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.inspection import DecisionBoundaryDisplay

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from IPython.display import display, Markdown, Latex

# Baseline Imports
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

import torch

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier, AutoTabPFNRegressor


# table for the encoding of the resistance testing into three classes: "susceptible", "intermediate-level resistant", "high-level resistant" with lower and upper thresholds

thresholds = [
    [3, 15],    # FPV
    [3, 15],    # ATV
    [3, 15],    # IDV
    [9, 55],    # LPV
    [3, 6],     # NFV
    [3, 15],    # SQV
    [2, 8],     # TPV
    [10, 90],   # DRV
    [5, 25],    # X3TC
    [2, 6],     # ABC
    [3, 15],    # AZT
    [1.5, 3],   # D4T
    [1.5, 3],   # DDI
    [1.5, 3],   # TDF
    [3, 10],    # EFV
    [3, 10],    # NVP
    [3, 10],    # ETR
    [3, 10],    # RPV
]

# Define row and column names
index = ["FPV","ATV","IDV","LPV","NFV","SQV","TPV","DRV",
         "3TC","ABC","AZT","D4T","DDI","TDF",
         "EFV","NVP","ETR","RPV"]
columns = ["lower", "upper"]

# Create DataFrame
cutoff_df = pd.DataFrame(thresholds, index=index, columns=columns)

#print(cutoff_df.head())

# Reading in and processing high quality File

df = pd.read_csv(r"data/PI_DataSet.txt", sep='\t')
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
                        "AUC ROC",
                        "Time",
                        "AUC RF",
                        "AUC XGB",
                        "AUC CatB"])


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

    # encoding the levels of susceptibility as 0 for susceptible, 1 as partly resistant and 2 as compeltly resistant
    dataframe.loc[dataframe[drug] < cutoff_df.loc[drug, "lower"], drug + "_level"] = 0
    dataframe.loc[dataframe[drug] >= cutoff_df.loc[drug, "upper"], drug + "_level"] = 2
    dataframe.loc[(dataframe[drug] >= cutoff_df.loc[drug, "lower"]) & (dataframe[drug] < cutoff_df.loc[drug, "upper"]), drug + "_level"] = 1

    #print(dataframe.head())

    X, y = dataframe.drop([drug, drug + "_level"], axis=1), np.array(dataframe[drug + "_level"])

    print(X)

    X_trafo = enc.transform(X).toarray()

    print(X_trafo.shape)

    #print(y)
    X_train, X_test, y_train, y_test = train_test_split(X_trafo, y, test_size=0.33, random_state=42)

    start_time = time.time()
    # Train and evaluate TabPFN
    y_pred = TabPFNClassifier(random_state=42, ignore_pretraining_limits=True).fit(X_train, y_train).predict_proba(X_test)

    taken_time = time.time() - start_time

    # Calculate ROC AUC (handles both binary and multiclass)
    score = roc_auc_score(y_test, y_pred if len(np.unique(y)) > 2 else y_pred[:, 1], multi_class='ovr')
    print(f"TabPFN ROC AUC: {score:.4f}")


    # Calculate ROC AUC (handles both binary and multiclass)
    score = roc_auc_score(y_test, y_pred if len(np.unique(y)) > 2 else y_pred[:, 1], multi_class='ovr')
    print(f"TabPFN ROC AUC: {score:.4f}")
    '''
    # Calculate ROC AUC (handles both binary and multiclass)
    score = roc_auc_score(y_test, y_pred if len(np.unique(y)) > 2 else y_pred[:, 1], multi_class='ovo', average='macro')
    print(f"TabPFN ROC AUC: {score:.4f}")'''

    # Define models
    models = [
        #('TabPFN', TabPFNClassifier(random_state=42)),
        ('RandomForest', RandomForestClassifier(random_state=42)),
        ('XGBoost', XGBClassifier(random_state=42)),
        ('CatBoost', CatBoostClassifier(random_state=42, verbose=0))
    ]

    # Calculate scores
    scoring = 'roc_auc_ovr' if len(np.unique(y)) > 2 else 'roc_auc'
    scores = {name: cross_val_score(model, X_trafo, y, cv=5, scoring=scoring, n_jobs=1, verbose=1).mean()
              for name, model in models}
    scores.update({'TabPFN':score})

    #saving the resulting statistics
    results = pd.concat([pd.DataFrame([[drug, score, taken_time, scores['RandomForest'], scores['XGBoost'], scores['CatBoost']]], columns=results.columns), results], ignore_index=True)

    for model, score in scores.items():
        print(model + ": " + str(score))

results.to_csv("Results_drugs_hq")