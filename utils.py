"""Utility Script for various predictive methods of TabPFN """

# Setup Imports
import pandas as pd
import numpy as np
import time
from sklearn.metrics import (
    precision_recall_curve,
    auc, average_precision_score
)
from sklearn.preprocessing import LabelBinarizer


THRESHOLDS = [
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
THRESHOLD_INDICES = ["FPV", "ATV", "IDV", "LPV", "NFV", "SQV", "TPV", "DRV",
                     "3TC", "ABC", "AZT", "D4T", "DDI", "TDF",
                     "EFV", "NVP", "ETR", "RPV", "BIC", "DTG", "EVG", "RAL"]

THRESHOLD_COLUMNS = ["lower", "upper"]


def get_thresholds():
    """
    Return a DataFrame of drug resistance score thresholds.

    Each row corresponds to a specific antiretroviral drug, and each column
    represents the lower and upper thresholds for interpreting resistance scores.

    The thresholds are commonly used to categorize resistance levels based on
    phenotype or genotype data.

    Returns:
        pandas.DataFrame: A DataFrame with 22 rows (each representing a drug) and
        two columns ('lower', 'upper'), containing threshold values for each drug.
    """

    return pd.DataFrame(THRESHOLDS, index=THRESHOLD_INDICES, columns=THRESHOLD_COLUMNS)




def get_classes(df, drug, mode="multiclass"):
    """
        Classify resistance scores for a given drug into binary or multiclass levels.

    Args:
        df (pandas.DataFrame): DataFrame containing resistance scores for various drugs.
        drug (str): The drug name (must match the threshold index).
        mode (str, optional): Classification mode. Must be either "multiclass" or "binary".
                              Defaults to "multiclass".

    Returns:
        numpy.ndarray: Array of classified resistance levels:
                       - In 'multiclass' mode: 0 = susceptible, 1 = intermediate, 2 = resistant.
                       - In 'binary' mode: 0 = susceptible, 1 = resistant.
    """

    lower, upper = get_thresholds().loc[drug, ["lower", "upper"]]

    if mode == "multiclass":
        conditions = [
            df[drug] < lower,
            df[drug] >= upper
        ]
        choices = [0, 2]
        default = 1
    elif mode == "binary":
        conditions = [df[drug] < lower]
        choices = [0]
        default = 1
    else:
        raise ValueError("mode must be either 'multiclass' or 'binary'")

    return np.select(conditions, choices, default=default)


def prc_auc_score(y_true, y_score, multiclass="raise"):

    if y_score.shape[1] == 0:
        tab_prec, tab_rec, thresholds = precision_recall_curve(y_true, y_score[:, 1])
        score_prc = auc(tab_rec, tab_prec)

    else:
        if multiclass == "raise":
            raise ValueError("multi_class must be in ('ovo', 'ovr')")
        elif multiclass == "ovo":
            pass
        elif multiclass == "ovr":

            label_binarizer = LabelBinarizer().fit(y_test)
            y_onehot_test = label_binarizer.transform(y_test)

            # print(y_test)
            # print(y_onehot_test)

            Y_test = y_onehot_test

            y_score = y_pred
            n_classes = 3

            # For each class
            precision = dict()
            recall = dict()
            average_precision = dict()
            for i in range(n_classes):
                precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i], y_score[:, i])
                average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

            # A "micro-average": quantifying score on all classes jointly
            precision["micro"], recall["micro"], _ = precision_recall_curve(
                Y_test.ravel(), y_score.ravel()
            )
            average_precision["micro"] = average_precision_score(Y_test, y_score, average="micro")


    tab_prec, tab_rec, thresholds = precision_recall_curve(y_true, y_score[:, 1])
    score_prc = auc(tab_rec, tab_prec)



    return score_prc






















