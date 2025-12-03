"""File for Data preprocessing of the Stanford HIV database"""


# Setup Imports
import pandas as pd
import numpy as np


# Thresholds for the classification of HIV drug resistances
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
        Function to return a DataFrame of drug resistance score thresholds

        :return: DataFrame containing resistance thresholds for each antiretroviral drug.
                 Each row corresponds to a drug, and the columns ('lower', 'upper')
                 represent the threshold values used to interpret resistance scores.
    """

    return pd.DataFrame(THRESHOLDS, index=THRESHOLD_INDICES, columns=THRESHOLD_COLUMNS)


def get_classes(df, drugs, mode="binary"):
    """
        Function to classify resistance scores for a given drug or list of drugs
        into binary or multiclass levels

        :param df: DataFrame containing resistance scores for various drugs
        :param drugs: Drug name or list of drugs (must match the threshold index)
        :param mode: Classification mode, either "multiclass" or "binary". Default is "binary"
        :return: DataFrame of categorical classes for the given drugs
                 - In 'multiclass' mode: 0 = susceptible, 1 = intermediate, 2 = resistant
                 - In 'binary' mode: 0 = susceptible, 1 = resistant
    """

    if type(drugs) != list:
        drugs = [drugs]

    classes = {}


    for drug in drugs:

        lower, upper = get_thresholds().loc[drug, ["lower", "upper"]]

        if mode == "multiclass":
            conditions = [
                df[drug] < lower,
                df[drug] >= upper,
                np.isnan(df[drug])
            ]
            choices = [0, 2, np.nan]
            default = 1
        elif mode == "binary":
            conditions = [
                df[drug] < lower,
                np.isnan(df[drug])]
            choices = [0, np.nan]
            default = 1
        else:
            raise ValueError("mode must be either 'multiclass' or 'binary'")

        classes.update({drug : np.select(conditions, choices, default=default)})

    return pd.DataFrame(classes)


def hq_hiv_loader(filename, drop_na=False, class_mode="binary"):
    """
        Function to load and preprocess high-quality HIV datasets from the Stanford HIV database

        :param filename: Filename of the target dataset to be processed
        :param drop_na: Whether to drop all rows containing NaNs in one or more targets. Default is False
        :param class_mode: Classification mode, either "multiclass" or "binary". Default is "binary"
        :return:
                 X (DataFrame): DataFrame of input features with shape (T, H),
                                where T is the number of examples and H is the number of features
                 Y (DataFrame): DataFrame of categorical classes with shape (T, L),
                                where T is the number of examples and L is the number of labels
                                - In 'multiclass' mode: 0 = susceptible, 1 = intermediate, 2 = resistant
                                - In 'binary' mode: 0 = susceptible, 1 = resistant
                 drugs (list[str]): List of label names in the dataset
    """

    # Reading in and processing high quality File
    df = pd.read_csv(filename, sep='\t')

    # removing index and summary column
    df = df.iloc[:, 1:-1]

    # list of current drugs of the dataset
    drugs = [drug for drug in list(df.columns) if not drug.startswith("P")]

    #Filtering out drugs with less than 10 labels present
    unusable_drugs = [drug for drug in drugs if df[drug].count() <= 10]

    if len(unusable_drugs) > 0:
        df.drop(columns=unusable_drugs, inplace=True)

        drugs = [drug for drug in drugs if drug not in unusable_drugs]


    if drop_na:
        #dropping rows with na labels
        df.dropna(subset=drugs, inplace=True)

    # collecting all features
    X = df.drop(drugs, axis=1)

    #getting the classlabels from the laboratory values
    Y = get_classes(df, drugs, mode=class_mode)

    return X, Y, drugs
