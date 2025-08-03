"""This script calculates the ROC AUC for the prediction of TabPFN, Random Forest, XGBoost, and
CatBoost and saves time in a file for all drugs in the stanford database file"""

# Setup Imports
import pandas as pd
import numpy as np
import time

import utils

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


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Baseline Imports
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from catboost import CatBoostClassifier, CatBoostRegressor



from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier, AutoTabPFNRegressor




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
                                    "AUC PRC",
                                    "AUC ROC",
                                    "Time"])



    for drug in drugs:

        tmp_drugs = drugs.copy().remove(drug)

        #getting labels of only needed drug
        #dataframe = df.drop(tmp_drugs, axis=1)

        dataframe = dataframe.dropna(subset=[drug])

        #If no thresholds for drug available no prediction possible
        if drug not in utils.THRESHOLD_INDICES:
            results = pd.concat([pd.DataFrame([[drug, dataframe.shape[0], None, None, None,
                                                None, None]], columns=results.columns),
                                 results], ignore_index=True)
            continue


        # encoding the levels of susceptibility as 0 for susceptible, 1 as partly resistant and 2 as completly resistant
        y = utils.get_classes(dataframe, drug, mode="multiclass")


        X = dataframe.drop([drug], axis=1)



        X_trafo = enc.transform(X).toarray()



        #getting train test split
        X_train, X_test, y_train, y_test = train_test_split(X_trafo, y, test_size=0.33, random_state=42)

        #Timing TabPFN
        start_time = time.time()

        # Train and evaluate TabPFN
        y_pred = TabPFNClassifier(random_state=42, ignore_pretraining_limits=True).fit(X_train, y_train).predict_proba(X_test)

        taken_time = time.time() - start_time

        # Calculate ROC AUC (handles both binary and multiclass)
        score = roc_auc_score(y_test, y_pred if len(np.unique(y)) > 2 else y_pred[:, 1], multi_class='ovr')
        print(f"TabPFN ROC AUC: {score:.4f}")



        #saving the resulting statistics
        results = pd.concat([pd.DataFrame([[drug, X_trafo.shape[0], score, taken_time, scores['RandomForest'], scores['XGBoost'], scores['CatBoost']]], columns=results.columns), results], ignore_index=True)

        for model, score in scores.items():
            print(model + ": " + str(score))

    results.to_csv(output_file)

def main():

    '''
    files = [r"data/PI_DataSet.txt", r"data/INI_DataSet.txt", r"data/NRTI_DataSet.txt", r"data/NNRTI_DataSet.txt"]

    for file in files:
        running_models(file, "output/" + (file.split("/")[-1].strip(".txt") + "_results.csv"))
    '''
    file = r"../data/INI_DataSet.txt"

    running_models(file, "output/" + (file.split("/")[-1].strip(".txt") + "_results.csv"))
if __name__ == '__main__':
    main()