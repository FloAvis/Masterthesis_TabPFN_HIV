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



def save_multilabel(y_pred, y_true, label, k_folds=None, path="../prediction_results/"):
    """
    Script to save the prediction results to a file for later evaluation

    :param y_pred: Predicted labels
    :param y_true: true labels
    :param label: name for the file without .csv attachement
    :param path: path of directory where the file should be saved. Default
    :return: Saving predicted and true labels into file
    """

    if y_true.shape != y_pred.shape:
        raise Exception("True labels  do not match predicted labels: " + str(y_true.shape) + " != " + str(y_pred.shape))

    splt = label.split('/')[:-1]
    sub_filepath = '/'.join(splt)

    #print(sub_filepath)

    Path(path + sub_filepath).mkdir(parents=True, exist_ok=True)

    y_pred_new = y_pred.add_prefix("Pred_")

    if k_folds is not None:
        y_pred_new["kFolds"] = k_folds

    y_true_new = y_true.add_prefix("True_")

    y_true_new.reset_index(inplace=True, drop=True)

    df = pd.concat([y_true_new, y_pred_new], axis=1, sort=False)

    df.to_csv(path + label + ".csv")

def subset_acc(y_pred, y_true, nan_mode="warning"):
    """
    Calculates the subset accuracy for multilabel prediction

    :param y_pred: Predicted labels
    :param y_true: true labels
    :param nan_mode: mode determining what happens when NaN is encountered:
                        - "warning": message informing about presence of NaNs which always leads to negative result
                        - "ignore": ignoring the NaNs in calculation
    :return: calculated subset accuracy
    """

    acc = 0

    for i in range(y_pred.shape[0]):
        tmp = 1
        for j in range(y_pred.shape[1]):
            if np.isnan(y_true.iloc[i,j]):
                if nan_mode == "ignore":
                    continue
                elif nan_mode == "warning":
                    print("Warning: True label is Missing, example not determinable")
                    tmp = 0
                    break
                else:
                    print("Invalid NaN handling")
                    return
            else:
                if y_pred.iloc[i,j] != y_true.iloc[i,j]:
                    tmp = 0
                    break

        acc = acc + tmp

    acc = acc/y_pred.shape[0]

    return acc

def save_multilabel_proba(y_pred_probas, y_true, label, k_folds=None, path="../prediction_results/"):
    """
    Script to save the prediction probabilities for each label, binary only

    :param y_pred_probas: Probabilities of predictions
    :param y_true: true labels
    :param label: name for the file without .csv attachement
    :param k_folds: array with sequence of k_folds
    :param path: path of directory where the file should be saved. Default
    :return: Saving predicted probabilities of the positive labels and true labels into file
    """

    y_pred_dict = {}

    drugs = y_true.columns.values.tolist()

    for i, probas in enumerate(y_pred_probas):
        y_pred_dict.update({drugs[i]: probas[:, 1]})

    y_pred = pd.DataFrame(y_pred_dict)


    if y_true.shape != y_pred.shape:
        raise Exception("True labels  do not match predicted labels: " + str(y_true.shape) + " != " + str(y_pred.shape))


    splt = label.split('/')[:-1]
    sub_filepath = '/'.join(splt)

    #print(sub_filepath)

    Path(path + sub_filepath).mkdir(parents=True, exist_ok=True)

    y_pred_new = y_pred.add_prefix("Pred_Proba_")

    if k_folds is not None:
        y_pred_new["kFolds"] = k_folds


    y_true_new = y_true.add_prefix("True_")

    y_true_new.reset_index(inplace=True, drop=True)

    df = pd.concat([y_true_new, y_pred_new], axis=1, sort=False)

    df.to_csv(path + label + ".csv")


def calc_labels(y_pred):
    """
    Returns the labels for a given probability matrix

    :param y_pred_probas: Probabilities of predictions
    :return: np.ndarray of labels
    """

    y_pred_probas = np.array(y_pred)


    y_pred_new = np.zeros((y_pred_probas[0].shape[0], len(y_pred_probas)))

    #print(y_pred_new)

    for j, clas in enumerate(y_pred_probas):
        for i in range(clas.shape[0]):
            if clas[i, 1] >= 0.5:
                y_pred_new[i, j] = 1
            else:
                y_pred_new[i, j] = 0

    return y_pred_new


def save_ensemble(y_pred_ensemble, y_true, label, k_folds=None, path="../prediction_results/"):
    """
    Script to save the prediction probabilities for each label, binary only

    :param y_pred_probas: Probabilities of predictions
    :param y_true: true labels
    :param label: name for the file without .csv attachement
    :param k_folds: array with sequence of k_folds
    :param path: path of directory where the file should be saved. Default
    :return: Saving predicted probabilities of the positive labels and true labels into file
    """

    if np.array(y_pred_ensemble).shape[1:3] != y_true.shape:
        raise Exception("True labels  do not match predicted labels: " + str(y_true.shape) + " != " + str(np.array(y_pred_probas_ensemble).T.shape[1:3]))


    Path(path + label).mkdir(parents=True, exist_ok=True)

    drugs = y_true.columns.values.tolist()

    for j, y_pred_arr in enumerate(y_pred_ensemble):

        y_pred = pd.DataFrame(y_pred_arr, columns=drugs)

        y_pred_new = y_pred.add_prefix("Pred_")

        if k_folds is not None:
            y_pred_new["kFolds"] = k_folds

        y_true_new = y_true.add_prefix("True_")

        y_true_new.reset_index(inplace=True, drop=True)

        df = pd.concat([y_true_new, y_pred_new], axis=1, sort=False)

        df.to_csv(path + label + "/Ensemble_" + str(j) + ".csv")



def save_ensemble_proba(y_pred_probas_ensemble, y_true, label, k_folds=None, path="../prediction_results/"):
    """
    Script to save the prediction probabilities for each label, binary only

    :param y_pred_probas: Probabilities of predictions
    :param y_true: true labels
    :param label: name for the file without .csv attachement
    :param k_folds: array with sequence of k_folds
    :param path: path of directory where the file should be saved. Default
    :return: Saving predicted probabilities of the positive labels and true labels into file
    """

    if np.array(y_pred_probas_ensemble).T.shape[1:3] != y_true.shape:
        raise Exception("True labels  do not match predicted labels: " + str(y_true.shape) + " != " + str(np.array(y_pred_probas_ensemble).T.shape[1:3]))


    Path(path + label).mkdir(parents=True, exist_ok=True)

    for j, y_pred_probas in enumerate(y_pred_probas_ensemble):
        y_pred_dict = {}

        drugs = y_true.columns.values.tolist()

        for i, probas in enumerate(y_pred_probas):
            y_pred_dict.update({drugs[i]: probas[:, 1]})

        y_pred = pd.DataFrame(y_pred_dict)

        y_pred_new = y_pred.add_prefix("Pred_Proba_")

        if k_folds is not None:
            y_pred_new["kFolds"] = k_folds


        y_true_new = y_true.add_prefix("True_")

        y_true_new.reset_index(inplace=True, drop=True)

        df = pd.concat([y_true_new, y_pred_new], axis=1, sort=False)

        df.to_csv(path + label + "/Ensemble_probas_" + str(j) + ".csv")



def ensemble_cv_predict(ensemble, X, Y, cv, method="predict_proba"):

    y_pred = np.zeros((ensemble.n_jobs, X.shape[0], Y.shape[1], 2))  # (n_samples, n_labels, n_classes)

    X_arr = np.array(X)
    Y_arr = np.array(Y)

    for train_idx, test_idx in cv.split(X):
        ensemble.fit(X_arr[train_idx], Y_arr[train_idx])
        y_pred[:,test_idx] = ensemble.predict_proba(X_arr[test_idx])

    return y_pred


