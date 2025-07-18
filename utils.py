"""Utility Script for various predictive methods of TabPFN """

# Setup Imports
import pandas as pd
import numpy as np
import time

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


def prc_auc_score(moroc_auc scorede):


























