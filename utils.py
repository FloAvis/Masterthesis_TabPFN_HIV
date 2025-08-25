"""Utility Script for various predictive methods of TabPFN """

# Setup Imports
import pandas as pd
import numpy as np

from pathlib import Path


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




def get_classes(df, drugs, mode="binary"):
    """
        Classify resistance scores for a given drug or drugs into binary or multiclass levels.

    Args:
        df (pandas.DataFrame): DataFrame containing resistance scores for various drugs.
        drugs (str): The drugs name or list of drugs (must match the threshold index).
        mode (str, optional): Classification mode. Must be either "multiclass" or "binary".
                              Defaults to "binary".

    Returns:
        pandas.Dataframe: Dataframe of categorical classes of given drugs
                       - In 'multiclass' mode: 0 = susceptible, 1 = intermediate, 2 = resistant.
                       - In 'binary' mode: 0 = susceptible, 1 = resistant.
    """

    """if type(drugs) != list:
        drug_list = [drugs]
    else:
        drug_list = drugs
    print(type(drugs))
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

def prc_auc_score(y_true, y_score, multiclass="raise"):
    """
    Calculating AUC PRC for binary and multiclass setting. OVR multiclass setting
    was adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
    :param y_true: True labels
    :param y_score: Predicted labels
    :param multiclass: which mode of multiclass to use
    :return: prc score
    """
    if y_score.shape[1] == 2:
        tab_prec, tab_rec, thresholds = precision_recall_curve(y_true, y_score[:, 1])
        score_prc = auc(tab_rec, tab_prec)

    else:
        if multiclass == "raise":
            raise ValueError("multi_class must be in ('ovo', 'ovr')")
        elif multiclass == "ovo":
            pass
        elif multiclass == "ovr":

            label_binarizer = LabelBinarizer().fit(y_true)
            Y_test = label_binarizer.transform(y_true)

            # print(y_test)
            # print(y_onehot_test)


            n_classes = y_score.shape[1]

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

            # score_prc = average_precision_score(Y_test, y_score, average="micro")

            score_prc = auc(recall["micro"], precision["micro"])


    return score_prc



def save_results(y_pred, y_true, label, path="../prediction_results/"):
    """
    Script to save the prediction results to a file for later evaluation

    :param y_pred: Predicted probabilities of the classes
    :param y_true: true labels
    :param label: name for the file without .csv attachement
    :param path: path of directory where the file should be saved. Default
    :return: Saving predictions and true labels into file
    """

    if y_true.shape[0] != y_pred.shape[0]:
        raise Exception("True labels do not match predicted labels")

    splt = label.split('/')[:-1]
    sub_filepath = '/'.join(splt)

    #print(sub_filepath)

    Path(path + sub_filepath).mkdir(parents=True, exist_ok=True)

    data = {"True": y_true}

    for i, column in enumerate(y_pred.T):
        data.update( {str(i): column })

    df = pd.DataFrame(data)

    df.to_csv(path + label + ".csv")


















