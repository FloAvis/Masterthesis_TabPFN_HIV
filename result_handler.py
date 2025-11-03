"""Utility Script for various predictive methods of TabPFN """

# Setup Imports
import pandas as pd
import numpy as np


from pathlib import Path

def get_kfold(kf, X, Y):

    kfolds = np.zeros((Y.shape[0], 1))

    k = 0

    for _, test in kf.split(X, Y):
        for i in test:
            kfolds[i] = k
        k += 1

    return kfolds

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

